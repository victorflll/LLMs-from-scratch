"""Exporta somente registros existentes; não executa ou simula um modelo."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def export():
    path = ROOT / 'lesson/chapter4_visual_lesson.ipynb'
    notebook = json.loads(path.read_text())
    records = {}
    for index in (3, 7, 14, 17):
        cell = notebook['cells'][index]
        records[str(index + 1)] = {
            'cell': index + 1,
            'code': ''.join(cell['source']),
            'output': ''.join(''.join(o.get('text', [])) for o in cell.get('outputs', [])),
        }
    ids = ast.literal_eval(re.search(r'Tokens: (\[.*?\])', records['15']['output'])[1])
    generated = ast.literal_eval(re.search(r'IDs gerados: (\[.*?\])', records['18']['output'])[1])
    source = (ROOT / 'lesson/gpt.py').read_text()
    tree = ast.parse(source)
    code = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    config = ast.literal_eval(main.body[0].value)
    payload = {'source': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'ids': ids, 'generated': generated,
               'text': records['18']['output'].split('Texto:')[1].strip(),
               'records': records, 'code': code, 'config': config}
    (ROOT / 'web/v2/gpt_journey_data.js').write_text('window.JOURNEY_DATA = ' + json.dumps(payload, ensure_ascii=False, indent=2) + ';\n')
    return payload

if __name__ == '__main__':
    data = export()
    print(f"Exportados {len(data['records'])} registros; {len(data['generated'])} IDs. Nenhuma inferência executada.")
