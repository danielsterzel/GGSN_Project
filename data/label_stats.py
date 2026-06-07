from pathlib import Path
import numpy as np

text = Path('data/manifest.csv').read_text(encoding='utf-8-sig').strip().splitlines()[1:]
lengths = [len(line.split(',',1)[1]) for line in text]
print('samples', len(lengths))
print('min', min(lengths), 'max', max(lengths))
print('median,75,90,95,99', np.percentile(lengths, [50, 75, 90, 95, 99]))
