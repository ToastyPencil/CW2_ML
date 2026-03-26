import json
from pathlib import Path

from scripts.build_appendix_notebooks import _notebook_payload, _notebook_payload_to_appendix_tex


def test_notebook_payload_contains_explanatory_markdown() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _notebook_payload(
        title="TypiClust Original",
        sources=[root / "src/cw2_ml/al/typiclust.py"],
    )

    markdown_sources = [
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "markdown"
    ]
    merged = "\n".join(markdown_sources)

    assert "What this notebook contains" in merged
    assert "How to use it" in merged
    assert "src/cw2_ml/al/typiclust.py" in merged


def test_generated_notebook_is_valid_json(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _notebook_payload(
        title="TypiClust+Adaptive Modified",
        sources=[root / "src/cw2_ml/al/modified.py"],
    )

    notebook_path = tmp_path / "notebook.ipynb"
    notebook_path.write_text(json.dumps(payload), encoding="utf-8")

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in data["cells"])


def test_modified_notebook_summary_mentions_uncertainty_schedule() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _notebook_payload(
        title="TypiClust+Adaptive Modified",
        sources=[root / "src/cw2_ml/al/modified.py"],
        summary=(
            "What this notebook contains\n"
            "- The submitted implementation of the adaptive TypiClust modification.\n"
            "- The round-dependent novelty and uncertainty schedule used in the final coursework evaluation.\n"
        ),
    )

    markdown_sources = [
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "markdown"
    ]
    merged = "\n".join(markdown_sources)
    assert "uncertainty" in merged.lower()


def test_notebook_payload_to_appendix_tex_renders_markdown_and_code() -> None:
    payload = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Demo Notebook\n",
                    "What this notebook contains\n",
                    "- First point\n",
                    "- Second point\n",
                ],
            },
            {
                "cell_type": "code",
                "source": [
                    "print('hello')\n",
                ],
            },
        ]
    }

    rendered = _notebook_payload_to_appendix_tex("demo.ipynb", payload)

    assert r"\subsection*{Notebook Printout: \texttt{demo.ipynb}}" in rendered
    assert r"\begin{itemize}" in rendered
    assert r"\begin{lstlisting}[language=Python]" in rendered
    assert "print('hello')" in rendered
