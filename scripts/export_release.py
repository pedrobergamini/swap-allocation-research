"""Copy only the reviewed release file list to a new directory."""

import argparse
import hashlib
import shutil
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PARTS = {"work", "target", ".git", ".agents", ".claude", ".codex", ".venv", "__pycache__"}


def release_files(root):
    entries = (root / "release-files.txt").read_text().splitlines()
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("release file list must be nonempty and unique")
    files = []
    for entry in entries:
        relative = PurePosixPath(entry)
        if (
            not entry
            or relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != entry
            or PRIVATE_PARTS.intersection(relative.parts)
            or any(part.startswith(".env") for part in relative.parts)
        ):
            raise ValueError(f"unsafe release path: {entry}")
        source = root / relative
        if any(path.is_symlink() for path in [source, *source.parents] if path != root.parent):
            raise ValueError(f"symlink in release path: {entry}")
        if not source.is_file():
            raise ValueError(f"missing release file: {entry}")
        files.append((entry, source))
    return files


def export(root, destination):
    files = release_files(root)
    destination.mkdir(parents=True, exist_ok=False)
    checksums = []
    for relative, source in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        checksums.append(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {relative}")
    (destination / "SHA256SUMS.release").write_text("\n".join(checksums) + "\n")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwritten"
    )
    args = parser.parse_args()
    count = export(ROOT, args.output.absolute())
    print(f"Exported {count} reviewed files to {args.output}")


if __name__ == "__main__":
    main()
