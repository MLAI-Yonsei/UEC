from pathlib import Path
import json,os,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
import preparation_checks as checks
import prepare_alignment as prepare

class CheckTests(unittest.TestCase):
    def test_version_check_reads_metadata_without_running_input_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'bin/python';binary.parent.mkdir();binary.write_text('not an executable; metadata only')
            info=root/'lib/python3.11/site-packages/numpy-1.0.dist-info';info.mkdir(parents=True)
            (info/'METADATA').write_text('Name: numpy\nVersion: 1.0\n')
            self.assertEqual(checks.query_versions(binary,['numpy']),{'numpy':'1.0'})

    def test_version_check_supports_inherited_system_packages_with_local_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=root/'env';base=root/'base';(env/'bin').mkdir(parents=True);(env/'bin/python').write_text('metadata only')
            (base/'bin').mkdir(parents=True);(env/'pyvenv.cfg').write_text('home = '+str(base/'bin')+'\ninclude-system-site-packages = true\n')
            for prefix,name,version in [(env,'numpy','2.0'),(base,'numpy','1.0'),(base,'torch','3.0')]:
                d=prefix/'lib/python3.11/site-packages'/(name+'-'+version+'.dist-info');d.mkdir(parents=True);(d/'METADATA').write_text('Name: '+name+'\nVersion: '+version+'\n')
            self.assertEqual(checks.query_versions(env/'bin/python',['numpy','torch']),{'numpy':'2.0','torch':'3.0'})

    def test_runtime_policy_cannot_be_inherited_over_drift(self):
        p=dict(prompt='actual',revision='rev',versions={'numpy':'v'})
        self.assertEqual(checks.verify_model_policy('qwen',SimpleNamespace(policy=p),p),p)
        with self.assertRaisesRegex(ValueError,'runtime policy'):
            checks.verify_model_policy('harrier',SimpleNamespace(policy={**p,'prompt':'changed'}),p)

    def test_config_uses_independent_bound_producer_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=dict(delivery_version='v3',cache_scope='alignment_full',models={},consumed_files={},backend_files={})
            for key,name in [('input_path','uec_alignment_inputs.npz'),('observations_path','observations.jsonl'),('split_path','split.json')]:
                file=root/name;file.write_text('fixture');p[key]=str(file);p['consumed_files'][str(file)]=checks.sha(file)
            for model in ['qwen','harrier','gte']:
                file=root/model/'summary.json';file.parent.mkdir();record=dict(status='completed',model=model,policy=dict(revision='actual'))
                file.write_text(json.dumps(record));p['models'][model]=record;p['consumed_files'][str(file)]=checks.sha(file)
            file=root/'uec_input_provenance.json';file.write_text(json.dumps(p));checks.verified_provenance(root)
            p['models']['qwen']['policy']['revision']='forged';file.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'producer summary'):checks.verified_provenance(root)

    def test_legacy_runtime_is_required_and_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);author=root/'qwen_author_model.py';author.write_text('original author')
            models={m:dict(policy=dict(versions={'numpy':'v'},source_sha256=checks.sha(author))) for m in ['qwen','harrier','gte']};p=dict(models=models);config={}
            with self.assertRaisesRegex(FileNotFoundError,'runtime_policy_verification'):checks.bind_runtime_evidence(root,p,config)
            evidence=dict(timing='retrospective_after_cache_generation',model_runtime_versions={m:dict(interpreter='/fixture/python',versions={'numpy':'v'}) for m in models},additional_consumed_files={str(author):checks.sha(author)})
            file=root/'runtime_policy_verification.json';file.write_text(json.dumps(evidence))
            with patch.object(checks,'query_versions',return_value={'numpy':'v'}):checks.bind_runtime_evidence(root,p,config)
            self.assertEqual(config['runtime_verification_sha256'],checks.sha(file))
            author.write_text('drift')
            with patch.object(checks,'query_versions',return_value={'numpy':'v'}),self.assertRaisesRegex(ValueError,'runtime source'):checks.bind_runtime_evidence(root,p,{})

    def test_gpu_is_bound_by_uuid_and_supported_memory(self):
        def output(command,**kwargs):return '2, GPU-selected, 0, 49140, 0\n' if '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu' in command else ''
        with patch.object(prepare.subprocess,'check_output',side_effect=output):self.assertEqual(prepare.check_gpu(2),'GPU-selected')
        with patch.object(prepare.subprocess,'check_output',side_effect=lambda c,**k:output(c,**k).replace('49140','8000')):
            with self.assertRaisesRegex(RuntimeError,'memory'):prepare.check_gpu(2)

    def test_storage_settings_override_stale_home_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'UEC';root.mkdir();script=root.parent/'home-offload/storage-env.sh';script.parent.mkdir();script.write_text('export HF_HOME=/data2/claire/.cache/huggingface\nexport XDG_CACHE_HOME=/data1/claire/cache\nexport TMPDIR=/data1/claire/tmp\n')
            with patch.dict(os.environ,{'HF_HOME':'/home/stale/cache'}):env=prepare.storage_environment(root)
            self.assertEqual(env['HF_HOME'],'/data2/claire/.cache/huggingface');self.assertEqual(env['XDG_CACHE_HOME'],'/data1/claire/cache')

if __name__=='__main__':unittest.main()
