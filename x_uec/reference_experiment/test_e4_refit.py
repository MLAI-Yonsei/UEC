import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import e4_refit as e4
from test_covariance_ops import fixture,ORDER

def toy_data():
    maps,q,d=fixture()
    means={m:np.concatenate([q['native_means'][m],d['native_means'][m]]) for m in ORDER}
    var={m:np.concatenate([q['native_vars'][m],d['native_vars'][m]]) for m in ORDER}
    ids={m:np.array([f'obs:{i}' for i in range(12)]) for m in ORDER}
    observations=[dict(observation_id=f'obs:{i}',source_order=i,moment_side='query' if i<4 else 'document',canonical_text_sha256=str(i)) for i in range(12)]
    data=e4.E4Inputs(means,var,ids,copy.deepcopy(ids),observations,{}, {})
    split=e4.AlignmentSplit(np.array([0,3,6,7,10,11]),np.array([1,2,4,5,8,9]),np.array([],dtype=int),
        dict(moment_query_indices=[1,2],moment_document_indices=[4,5,8,9]))
    return data,split

class RefitTests(unittest.TestCase):
    def test_missing_cache_blocks_fit(self):
        with patch.object(e4.caia,'fit_caia',side_effect=AssertionError('must not fit')):
            self.assertEqual(e4.preflight({'provenance_path':'/nonexistent/uec_input_provenance.json'})['status'],'blocked_missing_inputs')

    def test_fit_only_rows_locked_config_and_gpa_continuation(self):
        data,split=toy_data();original=e4.caia.fit_caia
        calls=[]
        def record(*args,**kwargs):calls.append((args,kwargs));return original(*args,**kwargs)
        with patch.object(e4.caia,'fit_caia',record):result=e4.fit_e4(data,split,{})
        self.assertEqual(calls[0][1],dict(gamma=None,include_shuffled=False,r0_sweeps=3,max_sweeps=3))
        for m in ORDER:np.testing.assert_array_equal(calls[0][0][0][m],data.means[m][split.fit_indices])
        self.assertIn('gpa',result);self.assertIsNot(result['gpa'],result['r0'])
        self.assertEqual(set(result)&{'shuffled'},set())
        self.assertIn(result['gpa']['status'],['stationary_tolerance','budget_exhausted','stalled_not_stationary'])

    def test_raw_transport_no_mutation_and_both_means(self):
        data,split=toy_data();data.means['qwen']*=1.0002;saved=copy.deepcopy(data)
        result=e4.fit_e4(data,split,{})
        rows,geometry=e4.validate_e4_transport(data,split,result,{})
        self.assertEqual(len(rows),140)
        self.assertTrue(all(r['pass_equivalence'] for r in rows))
        self.assertEqual({r['term'] for r in rows},{'fused_mean_query','fused_mean_document',*e4.ops.TERMS})
        for m in ORDER:
            np.testing.assert_array_equal(data.means[m],saved.means[m]);np.testing.assert_array_equal(data.variances[m],saved.variances[m])

    def test_smoke_budget_nonapproval_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new'
            row=e4.run_with_deadline([sys.executable,'-c','import time; time.sleep(10)'],output,budget=.1)
            self.assertEqual(row['status'],'smoke_timeout');self.assertFalse(row['alignment_ready_for_handoff'])
            self.assertFalse((output/'maps').exists())
            with self.assertRaises(FileExistsError):e4.run_with_deadline([],output,budget=.1)

    def test_failed_map_not_exported_and_scope_flags(self):
        data,split=toy_data();result=e4.fit_e4(data,split,{})
        moments,geometry=e4.validate_e4_transport(data,split,result,{})
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);bad=copy.deepcopy(result);bad['caia']['status']='numerical_failure'
            source=output/'input.txt';source.write_text('actual input witness');data.checksums[str(source)]=e4.sha(source)
            with self.assertRaises(ValueError):e4.write_e4_outputs(output,data,split,bad,moments,geometry)
            self.assertFalse((output/'diagnostic_only').exists())
            e4.write_e4_outputs(output,data,split,result,moments,geometry)
            self.assertTrue((output/'diagnostic_only/gpa_maps.npz').exists())
            state=json.loads((output/'status.json').read_text())
            for key in ['alignment_ready_for_handoff','uec_fusion_executed','downstream_evaluated','theorem_extension_proved']:self.assertFalse(state[key])
            manifest=json.loads((output/'handoff_manifest.json').read_text())
            self.assertEqual(manifest['input_files'][str(source)],e4.sha(source))
            self.assertIn('run_covariance_validation.py',{Path(f).name for f in manifest['code_files']})

    def test_rejects_empty_or_failed_diagnostics_before_map_export(self):
        data,split=toy_data();result=e4.fit_e4(data,split,{})
        rows,geometry=e4.validate_e4_transport(data,split,result,{})
        failed=copy.deepcopy(rows);failed[0]['pass_equivalence']=False
        for moments,geo in [([],geometry),(failed,geometry),(rows,{})]:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):e4.write_e4_outputs(Path(tmp),data,split,result,moments,geo)
                self.assertFalse((Path(tmp)/'diagnostic_only').exists())

    def test_positive_gamma_without_covariance_guidance_is_flagged(self):
        data,split=toy_data();result=e4.fit_e4(data,split,{})
        rows,geometry=e4.validate_e4_transport(data,split,result,{})
        result['gamma']=.02;result['caia']['quality_flags'].append('no_cov_guidance')
        with tempfile.TemporaryDirectory() as tmp:
            e4.write_e4_outputs(Path(tmp),data,split,result,rows,geometry)
            self.assertFalse(json.loads((Path(tmp)/'status.json').read_text())['covariance_guidance_active'])

    def test_backend_pins_survive_identical_package_relocation(self):
        pins={str(Path('/old/extract')/f):e4.sha(Path(e4.__file__).parent/f) for f in ['caia.py','covariance_ops.py','run_covariance_validation.py']}
        e4.verify_backend_files(pins,pins)
        changed=dict(pins);changed['/old/extract/caia.py']='wrong'
        with self.assertRaises(ValueError):e4.verify_backend_files(changed,pins)

    def test_cli_failure_has_status_and_nonzero_exit(self):
        for provenance in [None,{'delivery_version':'wrong','cache_scope':'smoke_only'}]:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);prov=p/'prov.json'
                if provenance is not None:prov.write_text(json.dumps(provenance))
                config=p/'config.json';config.write_text(json.dumps(dict(provenance_path=str(prov))))
                proc=subprocess.run([sys.executable,str(Path(e4.__file__).with_name('run_e4_refit.py')),'run','--config',str(config),'--output',str(p/'out')],capture_output=True,text=True)
                self.assertNotEqual(proc.returncode,0)
                self.assertTrue((p/'out/status.json').exists())
                state=json.loads((p/'out/status.json').read_text())
                self.assertEqual(state['status'],'blocked_missing_inputs' if provenance is None else 'smoke_failed')

    def test_interrupted_progress_does_not_destroy_final_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out'
            command=[sys.executable,'-c',"from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{'); sys.exit(2)",str(p/'progress.json')]
            state=e4.run_with_deadline(command,p,budget=2)
            self.assertEqual(state['status'],'smoke_failed')
            self.assertTrue((p/'status.json').exists())

if __name__=='__main__':unittest.main()
