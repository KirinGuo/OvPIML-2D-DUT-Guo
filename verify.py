"""Check every distributed file using only the Python standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["file"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"File missing or incomplete: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"File changed: {path}")
    print(f"Verified {len(manifest['files'])} files ({manifest['bytes']:,} bytes).")


if __name__ == "__main__":
    main()
