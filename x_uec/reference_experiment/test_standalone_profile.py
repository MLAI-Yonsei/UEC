import json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import full_e4
sys.path.append(str(Path(__file__).resolve().parents[1]))

class ProfileReportTests(unittest.TestCase):
    def test_failed_smoke_preflight_is_recorded_as_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out'
            with patch('standalone.alignment.load_inputs',side_effect=ValueError('invalid test input')):
                with self.assertRaisesRegex(ValueError,'invalid test input'):
                    full_e4.run_full(dict(input_profile='standalone_v1',mode='smoke'),output)
            self.assertEqual(json.loads((output/'status.json').read_text())['mode'],'smoke')

if __name__=='__main__':unittest.main()
