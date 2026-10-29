#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
package_return.py | MOVER host.

Bundles ./outputs into mover_outputs.tar.gz and writes a SHA-256
manifest, so the return can be integrity-checked on arrival.

Nothing outside ./outputs is read or written, except the archive itself, which is
written to the package root. Raw MOVER rows are never copied: only the derived
outputs/ tree, which contains de-identified event-level derivatives of the kind
covered by the data-use agreement.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import tarfile

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "outputs"
ARCHIVE = PKG / "mover_outputs.tar.gz"


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = sorted(p for p in OUT.rglob('*') if p.is_file())
    if not files:
        raise SystemExit("outputs/ is empty — run --full first")
    manifest = {'generated': datetime.now().isoformat(), 'n_files': len(files), 'files': {}}
    for p in files:
        manifest['files'][str(p.relative_to(PKG))] = {
            'size': p.stat().st_size, 'sha256': sha256(p)}
    (OUT / 'RETURN_MANIFEST.json').write_text(json.dumps(manifest, indent=2))

    with tarfile.open(ARCHIVE, 'w:gz') as tar:
        tar.add(OUT, arcname='outputs')
    print(f"wrote {ARCHIVE}  ({ARCHIVE.stat().st_size/1e6:.2f} MB, {len(files)} files)")
    print(f"sha256 {sha256(ARCHIVE)}")


if __name__ == '__main__':
    main()
