import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import prepare_alignment as prepare
import run_alignment as run

class PreparationTests(unittest.TestCase):
    def test_prepare_fresh_outputs_delegates_all_actual_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'original';root.mkdir();output=Path(tmp)/'new_inputs'
            for name in ['.venv','.venv-reproduction']:
                p=root/name/'bin/python';p.parent.mkdir(parents=True);p.write_text('fixture interpreter')
            calls=[]
            def execute(command,**kwargs):calls.append((command,kwargs));return subprocess.CompletedProcess(command,0)
            with patch.object(prepare,'check_workspace'),patch.object(prepare,'check_gpu',return_value='GPU-fixture-2'),patch.object(prepare,'storage_environment',return_value={'TMPDIR':str(root)}),patch.object(prepare.subprocess,'run',side_effect=execute):
                prepare.prepare(root,output,gpu=2)
            stages=[call[0][2] for call in calls]
            self.assertEqual(stages,['data','model','model','model','assemble'])
            self.assertEqual([call[0][call[0].index('--model')+1] for call in calls[1:4]],['qwen','harrier','gte'])
            for command,kwargs in calls[1:4]:self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-fixture-2')
            self.assertIn('.venv-reproduction',calls[1][0][0]);self.assertIn('.venv/',calls[2][0][0])
            self.assertTrue(all(call[1]['check'] for call in calls))

    def test_missing_original_workspace_reports_real_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError,'original experiment workspace'):prepare.check_workspace(Path(tmp))

    def test_existing_preparation_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):prepare.prepare(Path(tmp),Path(tmp),gpu=1)

    def test_full_config_is_generated_from_actual_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p=dict(delivery_version='v3',cache_scope='alignment_full',backend_files={'caia.py':'actual'},models={m:dict(policy=dict(native_dim=d)) for m,d in [('qwen',1024),('harrier',640),('gte',768)]})
            from preparation_checks import sha
            p['consumed_files']={}
            for key,name in [('input_path','uec_alignment_inputs.npz'),('observations_path','observations.jsonl'),('split_path','split.json')]:
                f=folder/name;f.write_text('fixture');p[key]=str(f);p['consumed_files'][str(f)]=sha(f)
            for m,r in p['models'].items():
                r.update(status='completed',model=m);f=folder/m/'summary.json';f.parent.mkdir();f.write_text(json.dumps(r));p['consumed_files'][str(f)]=sha(f)
            (folder/'uec_input_provenance.json').write_text(json.dumps(p))
            config=run.config_from_inputs(folder)
            self.assertEqual(config['models'],{m:v['policy'] for m,v in p['models'].items()})
            self.assertEqual(config['backend_files'],p['backend_files']);self.assertEqual(config['mode'],'full')
            p['cache_scope']='smoke_only';(folder/'uec_input_provenance.json').write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'alignment_full'):run.config_from_inputs(folder)

if __name__=='__main__':unittest.main()
