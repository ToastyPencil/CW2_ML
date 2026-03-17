from pathlib import Path


def test_readme_contains_gpu_and_colab_instructions() -> None:
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "3070" in text
    assert "colab" in text
