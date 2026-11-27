import ast,json,shutil,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from standalone.storage import sha,write_json,write_jsonl,seal,code_hashes
from standalone.export import assemble,DIMS

class PortableLoaderTests(unittest.TestCase):
    def fixture(self,work):
        write_json(work/'preparation_config.json',dict(profile='smoke',probes=32,logical_batch=512,code_files=code_hashes()))
        data=work/'data';data.mkdir();rows=[dict(observation_id='query'),dict(observation_id='positive')]
        write_jsonl(data/'observations.jsonl',rows);seal(data,['observations.jsonl'],dict(profile='smoke'))
        data_sha=sha(data/'manifest.json')
        for m,d in DIMS.items():
            stage=work/'models'/m;gauss=stage/'gaussian';gauss.mkdir(parents=True)
            write_json(gauss/'summary.json',dict(coordinate_audit=dict(passed=True),parameter_gaussian_fitted_here=True))
            seal(gauss,['summary.json'],{})
            mean=np.zeros((2,d));mean[:,0]=1
            for key,value in [('mean',mean),('variance',np.ones_like(mean)),('mean_ids',np.array(['query','positive'])),('variance_ids',np.array(['query','positive']))]:np.save(stage/(key+'.npy'),value)
            write_json(stage/'summary.json',dict(data_sha256=data_sha,encoder_hash_before='fixed',encoder_hash_after='fixed',gaussian_manifest_sha256=sha(gauss/'manifest.json')))
            seal(stage,['summary.json','mean.npy','variance.npy','mean_ids.npy','variance_ids.npy','gaussian/manifest.json'],{})
        assemble(work)

    def test_whole_preparation_can_move_without_original_server_paths(self):
        from standalone.alignment import inspect_inputs,load_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();self.fixture(old)
            new=root/'moved';shutil.move(old,new)
            p=inspect_inputs(new/'inputs')
            ref=Path(__file__).parent/'reference_experiment';sys.path.insert(0,str(ref))
            try:data=load_inputs(dict(mode='smoke',provenance_path=str(new/'inputs/uec_input_provenance.json')))
            finally:sys.path.remove(str(ref))
            self.assertEqual(data.means['gte'].shape,(2,768))
            self.assertFalse(any(str(old) in key for key in data.checksums))
            with (new/'models/gte/variance_ids.npy').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):inspect_inputs(new/'inputs')

    def test_standalone_has_no_private_project_imports_or_absolute_paths(self):
        root=Path(__file__).parent/'standalone'
        forbidden={'caia_svd_pilot_v3','caia_svd_pilot_v2','caia_pair_pilot','gte_la','alignment_fma','uec_experiment','gte_frozen_uncertainty'}
        for path in root.glob('*.py'):
            source=path.read_text();tree=ast.parse(source)
            self.assertNotIn('/data2/claire/',source)
            for node in ast.walk(tree):
                names=[n.name.split('.')[0] for n in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
                self.assertFalse(set(names)&forbidden,(path,names))

if __name__=='__main__':unittest.main()
