"""Pinned public encoders and accepted v3 final-block feature paths."""
from importlib.metadata import version
from pathlib import Path
import hashlib
import torch
import torch.nn.functional as F
from . import math_qwen as qm,math_paths as pm,math_gte as gm
from .storage import sha

SPECS={
 'qwen':dict(model_id='Qwen/Qwen3-Embedding-0.6B',revision='b22da495047858cce924d27d76261e96be6febc0',dimension=1024,length=8192,prompt='Instruct: Retrieve semantically similar text\nQuery:'),
 'harrier':dict(model_id='microsoft/harrier-oss-v1-270m',revision='31de22b673913c7d658c0f03f792d77c2dcf8ebd',dimension=640,length=32768,prompt='Instruct: Retrieve semantically similar text\nQuery: '),
 'gte':dict(model_id='thenlper/gte-base',revision='c078288308d8dee004ab72c6191778064285ec0c',dimension=768,length=512,prompt='')}

def package_versions():
    return {p:version(p) for p in ['torch','transformers','sentence-transformers','numpy','huggingface-hub','safetensors','tokenizers']}

class ModelPath:
    def __init__(self,name,cache,device='cuda:0',offline=False,qwen_attention='sdpa'):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel,AutoTokenizer
        from sentence_transformers import SentenceTransformer
        self.name=name;self.spec=SPECS[name];self.device=torch.device(device);s=self.spec
        snapshot=Path(snapshot_download(s['model_id'],revision=s['revision'],cache_dir=str(cache),local_files_only=offline,
            allow_patterns=['*.json','*.safetensors','tokenizer*','vocab.txt','*.model','1_Pooling/*','2_Normalize/*']))
        if snapshot.name!=s['revision']:raise ValueError('model snapshot revision mismatch')
        self.snapshot=snapshot;self.asset_hashes={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.model')}
        self._features={}
        if name=='qwen':
            dtype=torch.float16 if self.device.type=='cuda' else torch.float32
            if self.device.type!='cuda' and qwen_attention=='flash_attention_2':raise ValueError('flash attention requires CUDA')
            self.encoder=AutoModel.from_pretrained(str(snapshot),local_files_only=True,torch_dtype=dtype,attn_implementation=qwen_attention).to(self.device)
            self.tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True);self.tokenizer.padding_side='left'
            self.base=self.encoder;self.block=self.base.layers[-1]
            if len(self.base.layers)!=28:raise ValueError('Qwen block count changed')
            self.weight=self.block.mlp.down_proj.weight.detach().double().clone();self.scale=self.base.norm.weight.detach().double().clone();self.eps=self.base.norm.variance_epsilon
            self.block.mlp.down_proj.register_forward_pre_hook(lambda m,a:self._store('h',a[0]))
            self.block.post_attention_layernorm.register_forward_pre_hook(lambda m,a:self._store('r',a[0]))
            if tuple(self.weight.shape)!=(1024,3072):raise ValueError('Qwen final projection changed')
        else:
            self.encoder=SentenceTransformer(str(snapshot),device=str(self.device),local_files_only=True,
                model_kwargs={'torch_dtype':torch.float32,'attn_implementation':'sdpa'})
            self.encoder.max_seq_length=s['length'];self.base=self.encoder[0].auto_model;self.tokenizer=self.encoder.tokenizer
            if name=='harrier':
                if len(self.base.layers)!=18:raise ValueError('Harrier block count changed')
                self.block=self.base.layers[-1];self.weight=self.block.mlp.down_proj.weight.detach().double().clone()
                self.constants=dict(post_scale=1+self.block.post_feedforward_layernorm.weight.detach().double(),
                    final_scale=1+self.base.norm.weight.detach().double(),post_eps=self.block.post_feedforward_layernorm.eps,final_eps=self.base.norm.eps)
                self.block.mlp.down_proj.register_forward_pre_hook(lambda m,a:self._store('h',a[0]))
                self.block.pre_feedforward_layernorm.register_forward_pre_hook(lambda m,a:self._store('r',a[0]))
                if tuple(self.weight.shape)!=(640,2048) or not self.encoder[1].pooling_mode_lasttoken:raise ValueError('Harrier final path/pooling changed')
            else:
                self.block=self.base.encoder.layer[-1].output;self.weight=self.block.dense.weight.detach().double().clone()
                self.constants=dict(bias=self.block.dense.bias.detach().double(),gamma=self.block.LayerNorm.weight.detach().double(),beta=self.block.LayerNorm.bias.detach().double(),eps=self.block.LayerNorm.eps)
                def hook(module,args):self._store('H',args[0]);self._store('R',args[1])
                self.block.register_forward_pre_hook(hook)
                if tuple(self.weight.shape)!=(768,3072) or not self.encoder[1].pooling_mode_mean_tokens or self.tokenizer.padding_side!='right':raise ValueError('GTE final path/pooling changed')
        self.encoder.eval();self.encoder.requires_grad_(False);self.initial_hash=self.parameter_hash()
        self.policy=dict(model_id=s['model_id'],revision=s['revision'],tokenizer_revision=s['revision'],native_dim=s['dimension'],max_seq_length=s['length'],
            prompt=s['prompt'],actual_dtype=str(next(self.encoder.parameters()).dtype),actual_attention_backend=self.base.config._attn_implementation,
            padding_side=self.tokenizer.padding_side,truncation_side=self.tokenizer.truncation_side,
            pooling='last_valid_token' if name in ('qwen','harrier') else 'masked_mean',
            versions=package_versions(),final_output='FP64 final-path mean and exact native diagonal after L2; source encoder comparison checked')

    def _store(self,key,value):self._features[key]=value.detach()

    def parameter_hash(self):
        h=hashlib.sha256()
        for name,p in self.encoder.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def verify_unchanged(self):
        if self.parameter_hash()!=self.initial_hash:raise ValueError('encoder parameters changed')
        if any(sha(self.snapshot/name)!=expected for name,expected in self.asset_hashes.items()):raise ValueError('model asset changed during computation')

    @torch.no_grad()
    def capture(self,texts):
        self._features={};s=self.spec
        if self.name=='qwen':
            tokens=self.tokenizer([s['prompt']+t for t in texts],padding=True,truncation=True,max_length=s['length'],return_tensors='pt').to(self.device)
            hidden=self.encoder(**tokens,return_dict=True).last_hidden_state;ix=qm.last_indices(tokens['attention_mask']);b=torch.arange(len(texts),device=self.device)
            actual=F.normalize(hidden[b,ix],p=2,dim=1)
        else:
            tokens={k:v.to(self.device) for k,v in self.encoder.tokenize([s['prompt']+t for t in texts]).items()}
            actual=self.encoder(tokens)['sentence_embedding'];ix=qm.last_indices(tokens['attention_mask']);b=torch.arange(len(texts),device=self.device)
        if not torch.isfinite(actual).all():raise ValueError('nonfinite encoder output')
        if self.name=='gte':
            features=[dict(H=self._features['H'][i,:int(tokens['attention_mask'][i].sum())].double().cpu(),
                R=self._features['R'][i,:int(tokens['attention_mask'][i].sum())].double().cpu(),
                mask=tokens['attention_mask'][i,:int(tokens['attention_mask'][i].sum())].double().cpu()) for i in range(len(texts))]
        else:features=[dict(h=self._features['h'][i,ix[i]].double().cpu(),r=self._features['r'][i,ix[i]].double().cpu()) for i in range(len(texts))]
        self._features={};return features,actual.detach().double().cpu()

    def on_device(self,f):return {k:v.to(self.device) for k,v in f.items()}

    def forward(self,f):
        f=self.on_device(f)
        if self.name=='qwen':return qm.forward(self.weight,f['h'],f['r'],self.scale,self.eps)
        if self.name=='harrier':return pm.harrier_forward(self.weight,f['h'],f['r'],self.constants)
        return gm.split_forward(self.weight,f,self.constants)

    def diagonal(self,f,variance):
        f=self.on_device(f)
        if self.name=='qwen':return qm.output_diagonal(self.weight,f['h'],f['r'],self.scale,self.eps,variance)
        if self.name=='harrier':return pm.harrier_covariance(self.weight,f['h'],f['r'],self.constants,variance,False)[1]
        return pm.gte_diagonal(self.weight,f,self.constants,variance)

    def raw_vjp(self,features,coefficients,chunk=8):
        result=torch.zeros_like(self.weight)
        for start in range(0,len(features),chunk):
            fs=[self.on_device(f) for f in features[start:start+chunk]];a=coefficients[start:start+len(fs)].to(self.device)
            if self.name=='gte':result+=gm.raw_vjp(self.weight,fs,self.constants,a)
            else:
                h=torch.stack([f['h'] for f in fs]);r=torch.stack([f['r'] for f in fs])
                result+=qm.raw_vjp(self.weight,h,r,self.scale,self.eps,a) if self.name=='qwen' else pm.harrier_vjp(self.weight,h,r,self.constants,a)
        return result

    def coordinate_jacobian(self,features,coordinates):
        result=[]
        for start in range(0,len(features),8):
            fs=[self.on_device(f) for f in features[start:start+8]]
            if self.name=='gte':part=gm.coordinate_jacobian(self.weight,fs,self.constants,coordinates)
            else:
                h=torch.stack([f['h'] for f in fs]);r=torch.stack([f['r'] for f in fs])
                part=qm.coordinate_jacobian(self.weight,h,r,self.scale,self.eps,coordinates) if self.name=='qwen' else pm.harrier_coordinate_jacobian(self.weight,h,r,self.constants,coordinates)
            result.append(part)
        return torch.cat(result)
