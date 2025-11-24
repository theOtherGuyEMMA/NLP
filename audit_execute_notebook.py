import json
import traceback
from pathlib import Path

PRIMARY_NB = Path('Speech_Classification_Swahili_patched.ipynb')
FALLBACK_NB = Path('Speech_Classification_Swahili.ipynb')
NB_PATH = PRIMARY_NB if PRIMARY_NB.exists() else FALLBACK_NB
REPORT_PATH = Path('notebook_execution_report.txt')

HEAVY_IMPORT_TOKENS = [
    'import torch', 'from torch', 'import torchaudio', 'from torchaudio',
    'import transformers', 'from transformers',
    'import datasets', 'from datasets',
]

def strip_magics(code: str) -> str:
    cleaned = []
    for line in code.splitlines():
        ls = line.strip()
        if ls.startswith('%') or ls.startswith('!'):
            # Skip IPython magics and shell commands
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def requires_heavy_package(code: str) -> bool:
    lcode = code.lower()
    return any(tok in lcode for tok in HEAVY_IMPORT_TOKENS)

def main():
    if not NB_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NB_PATH}")

    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    cells = nb.get('cells', [])

    exec_globals = {
        '__name__': '__main__'
    }

    report_lines = []
    error_count = 0
    total_code_cells = 0

    for idx, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        total_code_cells += 1
        src = ''.join(cell.get('source', []))
        normalized = strip_magics(src)

        # Skip cells that import heavy packages not available
        skip_reason = None
        if requires_heavy_package(normalized):
            try:
                # Try to import key modules referenced
                if 'torch' in normalized:
                    import importlib
                    importlib.import_module('torch')
                if 'torchaudio' in normalized:
                    import importlib
                    importlib.import_module('torchaudio')
                if 'transformers' in normalized:
                    import importlib
                    importlib.import_module('transformers')
                if 'datasets' in normalized:
                    import importlib
                    importlib.import_module('datasets')
            except Exception as e:
                skip_reason = f"Missing heavy package: {e}"

        if skip_reason:
            report_lines.append(f"Cell {idx}: SKIPPED ({skip_reason}) | First line: {normalized.splitlines()[0] if normalized.splitlines() else ''}")
            continue

        try:
            exec(compile(normalized, f'<cell {idx}>', 'exec'), exec_globals)
            report_lines.append(f"Cell {idx}: OK | First line: {normalized.splitlines()[0] if normalized.splitlines() else ''}")
        except Exception:
            error_count += 1
            tb = traceback.format_exc()
            first_line = normalized.splitlines()[0] if normalized.splitlines() else ''
            report_lines.append(f"Cell {idx}: ERROR | First line: {first_line}\n{tb}")

    REPORT_PATH.write_text('\n\n'.join(report_lines), encoding='utf-8')
    print(f"Executed {total_code_cells} code cells, errors: {error_count}. Report saved to {REPORT_PATH}")

if __name__ == '__main__':
    main()