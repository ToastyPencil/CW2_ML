#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
REPORT_DIR = ROOT / "report"
APPENDIX_TEX_PATH = REPORT_DIR / "generated_notebook_appendix.tex"


NOTEBOOK_SPECS: dict[str, tuple[str, str, str, list[Path]]] = {
    "typiclust_original.ipynb": (
        "TypiClust Original",
        "What this notebook contains\n- The submitted implementation of the original TypiClust acquisition rule.\n- The exact source file used by the coursework pipeline.\n",
        "How to use it\n- Treat this notebook as a printout appendix for the report.\n- The code cell is copied directly from the project source so markers can inspect the implementation without opening the repository.\n",
        [
            ROOT / "src/cw2_ml/al/typiclust.py",
        ],
    ),
    "typiclust_modified.ipynb": (
        "TypiClust+Adaptive Modified",
        "What this notebook contains\n- The submitted implementation of the adaptive TypiClust modification.\n- The round-dependent novelty and uncertainty schedule used in the final coursework evaluation.\n",
        "How to use it\n- Treat this notebook as a printout appendix for the report.\n- The code cell is copied directly from the project source so markers can inspect the late-stage hybrid modification in one place.\n",
        [
            ROOT / "src/cw2_ml/al/modified.py",
        ],
    ),
}


def _source_lines(path: Path) -> list[str]:
    return [f"# File: {path.relative_to(ROOT)}\n", *path.read_text(encoding="utf-8").splitlines(keepends=True)]


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _markdown_source_to_tex(source: list[str]) -> str:
    text = "".join(source).strip()
    if not text:
        return ""

    lines = text.splitlines()
    blocks: list[str] = []
    paragraph_lines: list[str] = []
    in_list = False

    def _flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            blocks.append(" ".join(paragraph_lines) + "\n")
            paragraph_lines = []

    def _close_list() -> None:
        nonlocal in_list
        if in_list:
            blocks.append(r"\end{itemize}")
            in_list = False

    first = lines[0].strip()
    if first.startswith("#"):
        level = len(first) - len(first.lstrip("#"))
        title = _escape_latex(first[level:].strip())
        blocks.append(r"\subsection*{" + title + "}" if level == 1 else r"\paragraph*{" + title + "}")
        lines = lines[1:]

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            _flush_paragraph()
            _close_list()
            continue
        if stripped.startswith("- "):
            _flush_paragraph()
            if not in_list:
                blocks.append(r"\begin{itemize}")
                in_list = True
            blocks.append(r"\item " + _escape_latex(stripped[2:].strip()))
            continue
        _close_list()
        paragraph_lines.append(_escape_latex(stripped))

    _flush_paragraph()
    _close_list()
    return "\n".join(blocks).strip() + "\n"


def _code_source_to_tex(source: list[str]) -> str:
    code = "".join(source).rstrip()
    if not code:
        return ""
    return "\n".join(
        [
            r"\begin{lstlisting}[language=Python]",
            code,
            r"\end{lstlisting}",
            "",
        ]
    )


def _notebook_payload_to_appendix_tex(notebook_name: str, payload: dict) -> str:
    blocks = [r"\subsection*{Notebook Printout: \texttt{" + _escape_latex(notebook_name) + "}}"]
    for cell in payload["cells"]:
        if cell["cell_type"] == "markdown":
            rendered = _markdown_source_to_tex(cell["source"])
        elif cell["cell_type"] == "code":
            rendered = _code_source_to_tex(cell["source"])
        else:
            rendered = ""
        if rendered:
            blocks.append(rendered)
    return "\n".join(blocks).strip() + "\n"


def _notebook_payload(title: str, sources: list[Path], summary: str = "", usage: str = "") -> dict:
    if not summary:
        summary = (
            "What this notebook contains\n"
            "- The submitted implementation source copied directly from the project.\n"
            "- A printout-friendly view for the coursework appendix.\n"
        )
    if not usage:
        usage = (
            "How to use it\n"
            "- Read the markdown cells as guidance for the appendix.\n"
            "- Read the code cells as the exact source submitted with the repository.\n"
        )
    cells: list[dict] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                "Notebook printout for the coursework appendix. Each code cell below contains the submitted implementation source.\n",
            ],
        }
    ]
    if summary:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {summary}"],
            }
        )
    if usage:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {usage}"],
            }
        )
    for source_path in sources:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## `{source_path.relative_to(ROOT)}`\n"],
            }
        )
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(source_path),
            }
        )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    appendix_sections: list[str] = []
    for notebook_name, (title, summary, usage, sources) in NOTEBOOK_SPECS.items():
        payload = _notebook_payload(title, sources, summary=summary, usage=usage)
        (NOTEBOOK_DIR / notebook_name).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        appendix_sections.append(_notebook_payload_to_appendix_tex(notebook_name, payload))
    APPENDIX_TEX_PATH.write_text("\n\\clearpage\n\n".join(appendix_sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
