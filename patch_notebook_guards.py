import json
from pathlib import Path

NB_PATH = Path('Speech_Classification_Swahili.ipynb')
OUT_PATH = Path('Speech_Classification_Swahili_patched.ipynb')


def guard_block(var_name: str, body: str, message: str) -> str:
    lines = [
        f"if '{var_name}' not in globals() or {var_name} is None:",
        f"    print('{message}')",
        "else:",
    ]
    indented = ['    ' + l for l in body.splitlines()]
    return '\n'.join(lines + indented)

def bool_guard_block(flag_name: str, body: str, message: str) -> str:
    lines = [
        f"if not globals().get('{flag_name}', False):",
        f"    print('{message}')",
        "else:",
    ]
    indented = ['    ' + l for l in body.splitlines()]
    return '\n'.join(lines + indented)


def patch_cell_source(src: str) -> str:
    s = ''.join(src)
    lowered = s.lower()

    # Guard dataset loading failures (Datasets Hub)
    if 'load_dataset' in s and 'common_voice' in lowered:
        body = '\n'.join('    ' + line for line in s.splitlines())
        patched = (
            "try:\n"
            + body + "\n"
            + "except Exception as e:\n"
            + "    print(f'Falling back: could not load Common Voice ({e}). Using synthetic dataset.')\n"
            + "    import numpy as np, random\n"
            + "    sr = 16000\n"
            + "    num_labels = 5\n"
            + "    def synth_sample(duration=1.0, cls=0):\n"
            + "        t = np.linspace(0, duration, int(sr*duration), endpoint=False)\n"
            + "        freq = 200 + cls*100\n"
            + "        y = 0.1*np.sin(2*np.pi*freq*t)\n"
            + "        return {'audio': {'array': y, 'sampling_rate': sr}, 'label': cls}\n"
            + "    class SimpleDataset:\n"
            + "        def __init__(self, items): self.items = list(items)\n"
            + "        def map(self, func): return SimpleDataset([func(x) for x in self.items])\n"
            + "        def shuffle(self, seed=42):\n"
            + "            random.Random(seed).shuffle(self.items); return self\n"
            + "        def __getitem__(self, key):\n"
            + "            if isinstance(key, int):\n"
            + "                return self.items[key]\n"
            + "            if isinstance(key, slice):\n"
            + "                return SimpleDataset(self.items[key])\n"
            + "            if isinstance(key, str):\n"
            + "                return [x.get(key) for x in self.items]\n"
            + "            raise TypeError('Unsupported key type for SimpleDataset')\n"
            + "        def __iter__(self): return iter(self.items)\n"
            + "        def __len__(self): return len(self.items)\n"
            + "    ds_train = SimpleDataset([synth_sample(cls=i % num_labels) for i in range(100)])\n"
            + "    ds_val = SimpleDataset([synth_sample(cls=i % num_labels) for i in range(50)])\n"
        )
        return patched

    # Functions relying on ds_train -> add guard
    if 'ds_train' in s and ('def ' in s or 'map(' in s or 'for ' in s or 'shuffle(' in s):
        if 'wav2vec2' in lowered or 'facebook/wav2vec2-base' in lowered:
            body = s
            lines = [
                "if not globals().get('ENABLE_WAV2VEC2', False):",
                "    print('Skipping Wav2Vec2 section (disabled)')",
                "elif 'ds_train' not in globals() or ds_train is None:",
                "    print('Skipping cell: ds_train unavailable')",
                "else:",
            ]
            indented = ['    ' + l for l in body.splitlines()]
            return '\n'.join(lines + indented)
        # If the cell uses plotting or librosa, add inline imports with try/except
        needs_try = ('plt' in s) or ('seaborn' in lowered) or ('librosa' in lowered)
        if needs_try:
            import_lines = []
            if ('plt' in s) or ('matplotlib' in lowered):
                import_lines.append('import matplotlib.pyplot as plt')
            if ('seaborn' in lowered) or ('sns' in s):
                import_lines.append('import seaborn as sns')
            if 'librosa' in lowered:
                import_lines.append('import librosa')
                import_lines.append('import librosa.display')
            body_with_imports = '\n'.join(import_lines + s.splitlines())
            lines = [
                "if 'ds_train' not in globals() or ds_train is None:",
                "    print('Skipping cell: ds_train unavailable')",
                "else:",
                "    try:",
            ]
            indented = ['        ' + l for l in body_with_imports.splitlines()]
            tail = [
                "    except ModuleNotFoundError as e:",
                "        print(f'Skipping cell: missing package {e}')",
            ]
            return '\n'.join(lines + indented + tail)
        return guard_block('ds_train', s, 'Skipping cell: ds_train unavailable')

    # Cells relying on X_pad -> add guard
    if 'x_pad' in lowered:
        return guard_block('X_pad', s, 'Skipping cell: features X_pad unavailable')

    # Cells relying on model -> add guard
    if 'model' in s and ('evaluate' in lowered or 'embeddings' in lowered or 'predict' in lowered):
        return guard_block('model', s, 'Skipping cell: model unavailable')

    # Wav2Vec2 config needs num_labels
    if 'wav2vec2' in lowered or 'facebook/wav2vec2-base' in lowered:
        prelude = (
            "# Ensure num_labels exists\n"
            "num_labels = globals().get('num_labels', 5)\n"
        )
        guarded = bool_guard_block('ENABLE_WAV2VEC2', s, 'Skipping Wav2Vec2 section (disabled)')
        return prelude + guarded

    return s


def main():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    # Insert config cell enabling/disabling heavy sections
    config_cell = {
        'cell_type': 'code',
        'metadata': {"patched_config": True},
        'source': [
            "# Notebook runtime config (unconditional; avoid heavy imports here)\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import random\n",
            "from pathlib import Path\n",
            "\n",
            "# Basic runtime defaults\n",
            "ENABLE_WAV2VEC2 = globals().get('ENABLE_WAV2VEC2', False)\n",
            "RANDOM_SEED = globals().get('RANDOM_SEED', 42)\n",
            "np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)\n",
            "FIG_DIR = Path('figures'); FIG_DIR.mkdir(exist_ok=True)\n",
            "\n",
            "# Common globals used downstream\n",
            "num_labels = globals().get('num_labels', 5)\n",
            "ds_train = globals().get('ds_train', None)\n",
            "ds_val = globals().get('ds_val', None)\n",
            "print(f'Config ready. Wav2Vec2 enabled: {ENABLE_WAV2VEC2}, RANDOM_SEED={RANDOM_SEED}')\n",
        ],
    }
    nb.setdefault('cells', []).insert(1, config_cell)
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code' and not cell.get('metadata', {}).get('patched_config'):
            src = cell.get('source', [])
            patched = patch_cell_source(src)
            cell['source'] = [patched]
    OUT_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"Patched notebook written to {OUT_PATH}")


if __name__ == '__main__':
    main()