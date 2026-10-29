"""Download a pinned official snapshot; verify publisher hashes, retain partial files."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
DEST = ROOT / 'weights' / 'Qwen3.5-2B' / REVISION


def verify(path, entry):
    if not path.exists() or path.stat().st_size != entry['size']:
        return False
    h = hashlib.sha256() if 'lfs' in entry else hashlib.sha1()
    if 'lfs' not in entry:
        h.update(f"blob {entry['size']}\0".encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest() == (entry['lfs']['sha256'] if 'lfs' in entry else entry['blobId'])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=30) as response:
        meta = json.load(response)
    assert meta['sha'] == REVISION
    (DEST / 'source_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    # Small config/tokenizer files first, then weights. No repository Python code executed.
    entries = sorted((e for e in meta['siblings'] if e['rfilename'] != '.gitattributes'), key=lambda e: e['size'])
    for entry in entries:
        name = entry['rfilename']
        assert '/' not in name and '\\' not in name
        target = DEST / name
        if verify(target, entry):
            print('VERIFIED', name, flush=True)
            continue
        partial = target.with_name(name + '.part')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset != entry['size']:
            request = urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=45) as response:
                if offset and response.status != 206:
                    raise RuntimeError(f'Server did not honor resume for {name}; partial preserved.')
                if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected Content-Range; partial preserved.')
                last = time.monotonic()
                with partial.open('ab' if offset else 'wb') as f:
                    while block := response.read(4 * 1024**2):
                        f.write(block)
                        offset += len(block)
                        if time.monotonic() - last >= 20:
                            print(f'{name}: {offset / 1024**2:.0f}/{entry["size"] / 1024**2:.0f} MiB', flush=True)
                            last = time.monotonic()
        if not verify(partial, entry):
            raise RuntimeError(f'Publisher hash mismatch: {name}; partial preserved.')
        partial.replace(target)
        print('VERIFIED', name, flush=True)
    print('DOWNLOAD VERIFIED:', DEST, flush=True)


if __name__ == '__main__':
    main()
