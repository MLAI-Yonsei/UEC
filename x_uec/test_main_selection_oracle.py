"""Optional real-data oracle: run unmodified UEC main preprocessors themselves."""
import importlib.util,json,os,random,unittest
from pathlib import Path
import torch

@unittest.skipUnless(os.environ.get('XUEC_PUBLIC_CACHE'),'set XUEC_PUBLIC_CACHE to a downloaded public Hugging Face hub cache')
class MainSelectionOracle(unittest.TestCase):
    def test_all_selected_texts_and_order_equal_unmodified_main(self):
        import pyarrow.parquet as pq
        from standalone.assets import download_datasets,resolve_reference,REFERENCE
        from standalone.storage import sha
        root=Path(__file__).resolve().parents[1]
        protocol=json.loads((REFERENCE/'selection_protocol.json').read_text())
        for name,expected in protocol['source_files'].items():self.assertEqual(sha(root/name),expected)
        paths=download_datasets(os.environ['XUEC_PUBLIC_CACHE'],offline=True);pairs=resolve_reference(paths)
        class Rows:
            def __init__(self,table):self.table=table
            def __len__(self):return len(self.table)
            def __getitem__(self,i):return self.table.slice(i,1).to_pylist()[0]
        class Recorder:
            def __init__(self):self.calls=[]
            def __call__(self,texts,**kwargs):
                self.calls.append(list(texts));return {'input_ids':torch.zeros((len(texts),1),dtype=torch.long)}
        recorder=Recorder();state=random.getstate()
        try:
            random.seed(2025)
            for source,file,fn,ratio,count in [('ms_marco','ms_marco','preprocess_ms_marco',.05,3983),('snli','snli','preprocess_snli',.02,3775)]:
                spec=importlib.util.spec_from_file_location('uec_main_oracle_'+source,root/'utils/data_utils'/(file+'.py'))
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                actual=getattr(module,fn)(Rows(pq.read_table(paths[source])),recorder,batch_size=512,subset_ratio=ratio)
                expected=[r for r in pairs if r['source']==source]
                self.assertEqual(len(actual),count)
                self.assertEqual(recorder.calls[-2],[r['query'] for r in expected]);self.assertEqual(recorder.calls[-1],[r['positive'] for r in expected])
        finally:random.setstate(state)

if __name__=='__main__':unittest.main()
