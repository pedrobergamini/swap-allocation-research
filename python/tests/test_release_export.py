"""The public export must not follow paths into excluded local material."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "export_release", Path(__file__).resolve().parents[2] / "scripts/export_release.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "entry", ["../secret", "/etc/passwd", "work/states.json", ".env", "src/../secret"]
)
def test_export_rejects_private_or_escaping_paths(tmp_path, entry):
    (tmp_path / "release-files.txt").write_text(entry + "\n")
    with pytest.raises(ValueError, match="unsafe release path"):
        module.release_files(tmp_path)


def test_export_does_not_follow_a_symlink(tmp_path):
    (tmp_path / "private").write_text("private data")
    (tmp_path / "README.md").symlink_to(tmp_path / "private")
    (tmp_path / "release-files.txt").write_text("README.md\n")
    with pytest.raises(ValueError, match="symlink"):
        module.release_files(tmp_path)


def test_export_copies_only_reviewed_files_and_refuses_overwrite(tmp_path):
    root, destination = tmp_path / "repo", tmp_path / "public"
    root.mkdir()
    (root / "README.md").write_text("public")
    (root / "private.log").write_text("not for release")
    (root / "release-files.txt").write_text("README.md\n")
    assert module.export(root, destination) == 1
    assert (destination / "README.md").read_text() == "public"
    assert not (destination / "private.log").exists()
    with pytest.raises(FileExistsError):
        module.export(root, destination)
