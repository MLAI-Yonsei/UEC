import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import full_e4 as full
import e4_data
import e4_refit
from test_e4_refit import toy_data

class FullTests(unittest.TestCase):
    def test_full_selection_never_truncates_to_smoke(self):
        rows=[dict(observation_id=str(i),moment_side='query' if i%2==0 else 'document',canonical_text_sha256=str(i)) for i in range(3000)]
        split=dict(fit_indices=list(range(2500)),diagnostic_indices=list(range(2500,3000)),excluded_indices=[])
        selected=e4_data.select_scope(rows,split,'full')
        self.assertEqual(selected['selected_indices'],list(range(3000)))
        self.assertEqual(len(selected['fit_indices']),2500);self.assertEqual(len(selected['diagnostic_indices']),500)

    def test_full_fit_uses_existing_api_and_locked_full_limits(self):
        data,split=toy_data();original=e4_refit.caia.fit_caia;calls=[]
        def record(*args,**kwargs):calls.append(kwargs);return original(*args,**kwargs)
        with patch.object(e4_refit.caia,'fit_caia',record):e4_refit.fit_e4(data,split,dict(mode='full'))
        self.assertEqual(calls,[dict(gamma=None,include_shuffled=False,r0_sweeps=100,max_sweeps=300)])

    def test_full_loss_table_recomputes_gpa_covariance(self):
        data,split=toy_data();result=e4_refit.fit_e4(data,split,{})
        rows=full.evaluate_alignment_losses(data,split,result)
        self.assertEqual(len(rows),24)
        gpa=[r for r in rows if r['arm']=='gpa' and r['model_pair']=='all']
        self.assertTrue(all(r['cov_loss']>0 for r in gpa))
        self.assertTrue(all(abs(r['total_loss']-(r['mean_loss']+result['gamma']*r['cov_loss']))<1e-12 for r in rows))

    def test_bounded_pairs_preserve_side_and_exclude_same_text(self):
        observations=[dict(canonical_text_sha256=str(i)) for i in range(100000)]
        pairs=full.select_pairs(np.arange(50000),np.arange(50000,100000),observations,1000,42)
        self.assertEqual(len(pairs),1000);self.assertEqual(len(set(map(tuple,pairs))),1000)
        self.assertTrue(all(i<50000<=j for i,j in pairs))

    def test_readiness_requires_real_stationarity_and_numerics(self):
        data,split=toy_data();result=e4_refit.fit_e4(data,split,{})
        for arm in ['gpa','caia']:
            result[arm]['status']='stationary_tolerance';result[arm]['final_stationarity'].update(relative_residual=1e-9,relative_gap=1e-12)
        diagnostics=dict(geometry_passed=True,moment_passed=True,diagonal_completed=True)
        self.assertEqual(full.readiness(result,diagnostics),'alignment_ready_for_handoff')
        result['caia']['status']='budget_exhausted';self.assertEqual(full.readiness(result,diagnostics),'review_required')
        result['caia']['status']='stationary_tolerance';diagnostics['moment_passed']=False
        self.assertEqual(full.readiness(result,diagnostics),'numerical_failure')
        diagnostics['moment_passed']=True
        for invalid in [float('nan'),float('inf'),-1e-12]:
            result['caia']['final_stationarity'].update(relative_residual=invalid,relative_gap=1e-12)
            self.assertEqual(full.readiness(result,diagnostics),'review_required')

    def test_full_geometry_both_precisions_and_true_side_diagonal(self):
        data,split=toy_data()
        for i,row in enumerate(data.observations):row['original_triplet_id']=str(i//2)
        result=e4_refit.fit_e4(data,split,{})
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            ok,geometry=full.geometry_checks(data,split,result,output);self.assertTrue(ok);self.assertEqual(len(geometry),6)
            ok,moments=full.full_moment_checks(data,split,result,output);self.assertTrue(ok)
            self.assertEqual(len(moments),396)
            self.assertEqual({r['dtype'] for r in moments},{'float64','float32'})
            ok,summary=full.diagonal_checks(data,split,result,output);self.assertTrue(ok)
            self.assertTrue(all(r['absolute_max']<=1e-12 for r in summary if r['term']=='dot_mean' or r['condition']=='onehot_qwen_qwen'))
            losses=full.evaluate_alignment_losses(data,split,result)
            gp=geometry[0]['pairs'];dp=summary[0]['pairs']
            self.assertTrue(full.complete_diagnostics(losses,geometry,moments,summary,gp,dp))
            for partial in [(losses[:-1],geometry,moments,summary),(losses,geometry[:-1],moments,summary),(losses,geometry,moments[:-1],summary),(losses,geometry,moments,summary[:-1])]:
                self.assertFalse(full.complete_diagnostics(*partial,gp,dp))

    def test_late_export_failure_cannot_leave_approved_maps(self):
        data,split=toy_data();data.checksums={}
        data.provenance.update(cache_scope='alignment_full',uncertainty_source='frozen_teacher_v3_reused',models={m:dict(policy={}) for m in data.means})
        for i,row in enumerate(data.observations):row['original_triplet_id']=str(i//2)
        result=e4_refit.fit_e4(data,split,{})
        for arm in ['gpa','caia']:
            result[arm]['status']='stationary_tolerance';result[arm]['final_stationarity'].update(relative_residual=1e-9,relative_gap=1e-12)
        original=full.write_json
        def fail_manifest(path,value):
            if path.name=='handoff_manifest.json':raise OSError('simulated manifest write failure')
            return original(path,value)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new_run'
            with patch.object(full,'load_verified_inputs',return_value=data),patch.object(full,'make_full_split',return_value=split),patch.object(full,'audit_full_inputs',return_value={}),patch.object(full,'fit_e4',return_value=result),patch.object(full,'write_json',side_effect=fail_manifest):
                with self.assertRaisesRegex(OSError,'simulated'):full.run_full({},output)
            self.assertFalse((output/'maps').exists())

if __name__=='__main__':unittest.main()
