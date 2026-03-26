from pathlib import Path


def test_readme_contains_gpu_and_submission_instructions() -> None:
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "3070" in text
    assert "github" in text
    assert "overleaf" in text
