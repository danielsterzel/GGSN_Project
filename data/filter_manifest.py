import argparse
from pathlib import Path


def filter_manifest(src: Path, dst: Path, max_len: int = 32, image_col: str = 'image_path', text_col: str = 'text'):
    lines = src.read_text(encoding='utf-8-sig').splitlines()
    if not lines:
        raise SystemExit('Empty manifest')

    header = lines[0]
    rows = lines[1:]
    kept = []
    for row in rows:
        if not row.strip():
            continue
        parts = row.split(',', 1)
        if len(parts) != 2:
            continue
        text = parts[1]
        if len(text) <= max_len:
            kept.append(row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('\n'.join([header] + kept), encoding='utf-8')
    print(f'Wczytano {len(rows)} wierszy, zachowano {len(kept)} (max_len={max_len})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=str, default='data/manifest.csv')
    p.add_argument('--dst', type=str, default='data/manifest_filtered_32.csv')
    p.add_argument('--max-len', type=int, default=32)
    args = p.parse_args()
    filter_manifest(Path(args.src), Path(args.dst), max_len=args.max_len)
