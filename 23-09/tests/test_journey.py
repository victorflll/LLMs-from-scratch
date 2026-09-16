"""Verifica a origem dos registros usados na interface, sem executar inferência."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class JourneyDataTests(unittest.TestCase):
    def test_export_matches_notebook(self):
        raw = (ROOT / 'web/v2/gpt_journey_data.js').read_text()
        data = json.loads(raw.removeprefix('window.JOURNEY_DATA = ').removesuffix(';\n'))
        notebook = json.loads((ROOT / data['source']).read_text())
        self.assertEqual(data['sha256'], hashlib.sha256((ROOT / data['source']).read_bytes()).hexdigest())
        for record in data['records'].values():
            cell = notebook['cells'][record['cell'] - 1]
            self.assertEqual(record['code'], ''.join(cell['source']))
            self.assertEqual(record['output'], ''.join(''.join(o.get('text', [])) for o in cell['outputs']))
        self.assertEqual(data['generated'][:len(data['ids'])], data['ids'])
        self.assertEqual(len(data['generated']) - len(data['ids']), 10)
        self.assertEqual(data['records']['15']['output'].count('TransformerBlock'), data['config']['n_layers'])
        self.assertIn('(1, 4, 50257)', data['records']['15']['output'])

if __name__ == '__main__':
    unittest.main()
