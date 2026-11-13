"""Fetch the original official training data at a pinned revision, verifying SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for rel,expected in manifest['files'].items():
        path=ROOT/rel
        if not path.resolve().is_relative_to(ROOT/'data'):
            raise ValueError('Unsafe dataset path')
        if path.exists():
            if digest(path)!=expected['sha256']:
                raise ValueError('Existing dataset checksum differs: '+rel)
            print('Verified',rel)
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.download')
        url=f"https://raw.githubusercontent.com/TemporalSelfRewarding/TSR/{manifest['upstream_commit']}/{rel}"
        request=urllib.request.Request(url,headers={'User-Agent':'official-tsr-best-data'})
        with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as output:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                output.write(chunk)
        if digest(temporary)!=expected['sha256']:
            temporary.unlink()
            raise ValueError('Downloaded dataset checksum differs: '+rel)
        temporary.replace(path)
        print('Downloaded and verified',rel)


if __name__=='__main__':
    main()
