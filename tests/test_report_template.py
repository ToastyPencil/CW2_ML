from pathlib import Path


def test_report_template_mentions_algorithm_modification_and_statistics() -> None:
    content = Path("report/main.tex").read_text(encoding="utf-8")
    assert "Algorithm Modification" in content
    assert "Statistical Analysis" in content
    assert "GitHub" in content
    assert "generated_notebook_appendix.tex" in content
    assert "Your Name" not in content
    assert "Student ID" not in content
    assert "should state" not in content
    assert "should summarise" not in content
    assert "Attach notebook printouts" not in content
