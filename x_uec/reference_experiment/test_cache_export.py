import unittest
import numpy as np
import export_ag_cache as exporter

class ExportTests(unittest.TestCase):
    def test_source_row_orders_are_compared_before_role_export(self):
        anchors=[dict(row_id='ag:1',exact_hash='a'),dict(row_id='ag:2',exact_hash='b')]
        mean_rows=[dict(r) for r in anchors];metadata=[dict(r) for r in anchors]
        ids=exporter.verify_binding_rows(anchors,mean_rows,metadata)
        np.testing.assert_array_equal(ids,['ag:1:encoding_query','ag:2:encoding_query'])
        with self.assertRaises(ValueError):exporter.verify_binding_rows(anchors,mean_rows[::-1],metadata)
        with self.assertRaises(ValueError):exporter.verify_binding_rows(anchors,mean_rows,metadata[::-1])
        metadata[0]['exact_hash']='other'
        with self.assertRaises(ValueError):exporter.verify_binding_rows(anchors,mean_rows,metadata)

    def test_variance_sidecar_is_independently_validated(self):
        anchors=[dict(row_id='ag:1',exact_hash='a'),dict(row_id='ag:2',exact_hash='b')]
        sidecar=dict(ids=['ag:1','ag:2'],hashes=['a','b'],shape=[2,8],model='qwen',source='teacher',original_weight_hash='checkpoint')
        expected=['ag:1:encoding_query','ag:2:encoding_query']
        np.testing.assert_array_equal(exporter.verify_variance_rows(anchors,sidecar,'qwen',(2,8),'checkpoint'),expected)
        for key,replacement in [('ids',['ag:2','ag:1']),('hashes',['b','a']),('model','gte'),('source','prior'),('shape',[2,7]),('original_weight_hash','other')]:
            invalid=dict(sidecar);invalid[key]=replacement
            with self.subTest(key=key),self.assertRaises(ValueError):
                exporter.verify_variance_rows(anchors,invalid,'qwen',(2,8),'checkpoint')

if __name__=='__main__':unittest.main()
