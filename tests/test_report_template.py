from pathlib import Path


def test_report_template_mentions_required_sections() -> None:
    content = Path("report/main.tex").read_text(encoding="utf-8")
    for section in [
        "Introduction",
        "Methodology",
        "Results",
        "Statistical Analysis",
        "Conclusion",
        "Algorithm Modification",
    ]:
        assert section in content
