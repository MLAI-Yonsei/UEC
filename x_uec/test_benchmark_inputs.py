"""Catch cache identity/policy drift and unsealed portable benchmark inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from standalone.storage import sha, verify


def module():
    spec=importlib.util.spec_from_file_location('benchmark_inputs',Path(__file__).parent/'tools/benchmark_inputs.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fp(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fixture(root):
    row=dict(row_id='source:1',text='e\u0301\r\n two  spaces',exact_hash=hashlib.sha256('e\u0301\r\n two  spaces'.encode()).hexdigest())
    policy=dict(model_id='pinned/model',revision='abc',prompt='Query:',native_dim=2)
    cache_rows=[dict(row_id=row['row_id'],exact_hash=row['exact_hash'],cache_key=hashlib.sha256((row['text']+fp(policy)).encode()).hexdigest())]
    contract=dict(rows=cache_rows,encoder_config=policy,encoder_fingerprint=fp(policy),shape=[1,2])
    np.save(root/'array.npy',np.array([[3,4]],dtype=np.float32))
    (root/'meta.json').write_text(json.dumps(dict(complete=True,contract=contract,contract_fingerprint=fp(contract))))
    (root/'rows.json').write_text(json.dumps(cache_rows))
    ref=dict(array=dict(path=str(root/'array.npy'),key=None,shape=[1,2],dtype='float32',sha256=sha(root/'array.npy')),meta_path=str(root/'meta.json'),rows_path=str(root/'rows.json'),encoder_config=policy,files={str(root/n):sha(root/n) for n in ['array.npy','meta.json','rows.json']})
    return row,ref


class BenchmarkInputsTests(unittest.TestCase):
    def test_certified_cache_export_preserves_ids_and_normalizes_fp64(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            row,ref=fixture(Path(d));x=m.load_certified_cache(ref,[row])
            self.assertEqual(x.dtype,np.float64);np.testing.assert_array_equal(x,np.array([[.6,.8]]))
            self.assertEqual(m.canonical_text_sha256(row['text']),hashlib.sha256('é\n two  spaces'.encode()).hexdigest())
            for rows in [[dict(row,row_id='different')],[dict(row,text='changed')]]:
                with self.assertRaises(ValueError):m.load_certified_cache(ref,rows)
            ref['encoder_config']['revision']='wrong'
            with self.assertRaises(ValueError):m.load_certified_cache(ref,[row])

    def test_changed_sidecar_or_array_is_rejected(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,ref=fixture(root)
            (root/'rows.json').write_text('[]')
            with self.assertRaises(ValueError):m.load_certified_cache(ref,[row])
            row,ref=fixture(root);np.save(root/'array.npy',np.ones((1,2),dtype=np.float32))
            with self.assertRaises(ValueError):m.load_certified_cache(ref,[row])

    def test_export_seals_protocol_and_independent_identity(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,ref=fixture(root);task=root/'data/Tiny';task.mkdir(parents=True)
            (task/'support.jsonl').write_text(json.dumps(row)+'\n');(task/'test.jsonl').write_text(json.dumps(dict(row,row_id='test:1'))+'\n')
            labels=dict(support=[1],test=[1],class_order=[1],support_positions=[[0]],probe=dict(max_iter=100))
            (task/'labels.json').write_text(json.dumps(labels))
            manifest=dict(settings=dict(kind='Classification',repo='public/data',revision='123'),sources={},files={n:sha(task/n) for n in ['support.jsonl','test.jsonl','labels.json']})
            (task/'manifest.json').write_text(json.dumps(manifest))
            test_ref=json.loads(json.dumps(ref));meta=json.loads((root/'meta.json').read_text());meta['contract']['rows'][0]['row_id']='test:1';meta['contract_fingerprint']=fp(meta['contract'])
            (root/'test_meta.json').write_text(json.dumps(meta));(root/'test_rows.json').write_text(json.dumps(meta['contract']['rows']));test_ref.update(meta_path=str(root/'test_meta.json'),rows_path=str(root/'test_rows.json'));test_ref['files']={str(root/n):sha(root/n) for n in ['array.npy','test_meta.json','test_rows.json']}
            index={'Tiny':{'support':{'qwen':ref},'test':{'qwen':test_ref}}}
            (root/'index.json').write_text(json.dumps(index));(root/'registry.json').write_text(json.dumps([dict(runtime_task_id='Tiny',task_type='Classification')]))
            out=root/'portable';m.export_legacy(root/'index.json',root/'registry.json',root/'data',out)
            manifest=verify(out);self.assertEqual(manifest['metadata']['tasks'],['Tiny']);verify(out/'Tiny')
            task_data=json.loads((out/'Tiny/task.json').read_text());self.assertEqual(task_data['labels'],labels);self.assertEqual(task_data['roles']['test'][0]['row_id'],'test:1')
            self.assertIn('canonical_text_sha256',task_data['roles']['support'][0]);self.assertTrue((out/'Tiny/native.npz').is_file())
            (out/'Tiny/task.json').write_text('{}')
            with self.assertRaises(ValueError):verify(out)

    def test_public_reconstruction_uses_source_order_and_checks_labels(self):
        m=module()
        spec=dict(settings=dict(kind='Classification',train='train',split='test',text='text'),labels=dict(support=[1,0],test=[0],class_order=[0,1],support_train_indices=[[2,0]],support_positions=[[1,0]]))
        raw=dict(train=[dict(text='zero',label=1),dict(text='unused',label=0),dict(text='two',label=0)],test=[dict(text='test',label=0)])
        roles=m.reconstruct_roles('Tiny',spec,raw)
        self.assertEqual([r['row_id'] for r in roles['support']],['Tiny:train:0','Tiny:train:2'])
        self.assertEqual([r['text'] for r in roles['support']],['zero','two'])
        raw['test'][0]['label']=1
        with self.assertRaisesRegex(ValueError,'labels'):m.reconstruct_roles('Tiny',spec,raw)

    def test_reference_identity_rejects_reordered_or_changed_public_text(self):
        m=module();rows=[dict(row_id='id',text='known',exact_hash=hashlib.sha256(b'known').hexdigest())]
        reference={'texts':[['id','7117fff2d0fd294462b3c802b7cb8753579f23f3949b927370f550ae1e32d394']]}
        # Literal above deliberately differs: reconstruction must fail closed.
        with self.assertRaisesRegex(ValueError,'identity'):m.check_reference({'texts':rows},reference)
        reference['texts'][0][1]=hashlib.sha256(b'known').hexdigest();m.check_reference({'texts':rows},reference)
        rows[0]['text']='changed'
        with self.assertRaisesRegex(ValueError,'hash'):m.check_reference({'texts':rows},reference)

    def test_sts_public_reconstruction_checks_pair_identity_and_scale(self):
        m=module();raw={'test/en.jsonl.gz':[dict(id='pair:1',sentence1='left',sentence2='right',score=2.5)]}
        spec=dict(settings=dict(kind='STS',legacy='sts22'),pairs=[dict(id='pair:1',left_index=0,right_index=1,gold=2.5)])
        roles=m.reconstruct_roles('STS22.v2',spec,raw)
        self.assertEqual([r['text'] for r in roles['texts']],['left','right'])
        raw['test/en.jsonl.gz'][0]['score']=3.5
        with self.assertRaisesRegex(ValueError,'pair'):m.reconstruct_roles('STS22.v2',spec,raw)

    def test_native_mean_uses_final_path_and_checks_encoder_center(self):
        import torch
        m=module()
        class TinyModel:
            name='gte'
            spec=dict(prompt='old')
            mismatch=False
            def capture(self,texts):
                actual=torch.tensor([[.6,.8]]*len(texts),dtype=torch.float64)
                if self.mismatch:actual[:,0]=.4
                return [dict(value=torch.tensor([.6,.8],dtype=torch.float64)) for _ in texts],actual
            def forward(self,features):return None,features['value']
        model=TinyModel();values,audit=m.encode_mean_rows(model,[dict(text='raw')],prompt='role:',physical_batch=1)
        np.testing.assert_array_equal(values,[[.6,.8]]);self.assertEqual(audit['max_encoder_center_abs_error'],0.);self.assertEqual(model.spec['prompt'],'role:')
        model.mismatch=True
        with self.assertRaisesRegex(ValueError,'center'):m.encode_mean_rows(model,[dict(text='raw')],prompt='role:')

    def test_runtime_policy_rejects_padding_pooling_or_revision_drift(self):
        from types import SimpleNamespace
        m=module();model=SimpleNamespace(name='gte',spec=dict(model_id='public/model',revision='rev',length=512,dimension=2),policy=dict(tokenizer_revision='rev',padding_side='right',truncation_side='right',pooling='masked_mean'))
        expected=dict(model_id='public/model',revision='rev',tokenizer_revision='rev',max_seq_length=512,native_dim=2,actual_padding_side='right',actual_truncation_side='right',pooling=dict(pooling_mode_mean_tokens=True,pooling_mode_lasttoken=False))
        m.check_encoding_policy(model,expected)
        for key,value in [('revision','wrong'),('actual_padding_side','left'),('pooling',dict(pooling_mode_mean_tokens=False,pooling_mode_lasttoken=True))]:
            with self.assertRaisesRegex(ValueError,'policy'):m.check_encoding_policy(model,dict(expected,**{key:value}))

    def test_public_resume_rejects_mode_backend_batch_and_producer_drift(self):
        m=module();request=m.public_request('/tmp/cache','cuda:0',False,'sdpa',8)
        metadata=dict(mode='public',request=request)
        m.check_public_request(metadata,request)
        for key,value in [('qwen_attention','flash_attention_2'),('physical_batch',4),('implementation_sha256','changed')]:
            with self.assertRaisesRegex(ValueError,'request'):m.check_public_request(metadata,dict(request,**{key:value}))
        with self.assertRaisesRegex(ValueError,'mode'):m.check_public_request(dict(metadata,mode='legacy'),request)

    def test_suite_boundary_rejects_selfsealed_wrong_task_and_protocol(self):
        import gzip
        from unittest.mock import patch
        from standalone.storage import write_json,seal
        m=module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref=root/'ref';ref.mkdir();names=['SCIDOCS','LegalBenchCorporateLobbying','PoemSentimentClassification','TweetTopicSingleClassification','STSBenchmark','STS22.v2'];specs={};ids={}
            row=dict(row_id='one',text='text',exact_hash=hashlib.sha256(b'text').hexdigest(),canonical_text_sha256=hashlib.sha256(b'text').hexdigest())
            for name in names:
                specs[name]=dict(settings=dict(kind='STS'),registry={},pairs=[dict(id='pair',left_index=0,right_index=0,gold=1.)],policies={'texts':{key:dict(native_dim=2) for key in ['qwen','harrier','gte']}});ids[name]={'texts':[['one',row['exact_hash']]]}
            write_json(ref/'protocol.json',dict(tasks=specs));(ref/'row_ids.json.gz').write_bytes(gzip.compress(json.dumps(ids).encode()));seal(ref,['protocol.json','row_ids.json.gz'],dict(cache_index_sha256='cache',source_manifests={n:'source' for n in names}))
            def build(folder,wrong_task=False,wrong_pair=False):
                files=[]
                for name,spec in specs.items():
                    target=folder/name;target.mkdir(parents=True);record=dict(task='wrong' if wrong_task else name,kind='STS',source=spec['settings'],registry={},roles={'texts':[row]},pairs=[dict(spec['pairs'][0],gold=2.)] if wrong_pair else spec['pairs'])
                    write_json(target/'task.json',record);np.savez(target/'native.npz',**{'texts__'+key:np.array([[.6,.8]]) for key in ['qwen','harrier','gte']});write_json(target/'provenance.json',dict(cache_policies={key:{'texts':spec['policies']['texts'][key]} for key in ['qwen','harrier','gte']},source_manifest_sha256='source'));seal(target,['task.json','native.npz','provenance.json'],dict(source_manifest_sha256='source',cache_index_sha256='cache'));files.extend(name+'/'+f for f in ['task.json','native.npz','provenance.json','manifest.json'])
                write_json(folder/'provenance.json',dict(source_mode='explicit certified legacy cache import'));seal(folder,files+['provenance.json'],dict(tasks=names,schema_version=1,binding=dict(cache_index_sha256='cache')))
            with patch.object(m,'REFERENCE',ref):
                build(root/'good');self.assertEqual(m.validate_benchmark_suite(root/'good')['tasks'],names)
                for name,options in [('task',dict(wrong_task=True)),('protocol',dict(wrong_pair=True))]:
                    build(root/name,**options)
                    with self.assertRaises(ValueError):m.validate_benchmark_suite(root/name)

if __name__=='__main__':unittest.main()
