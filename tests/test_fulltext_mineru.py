"""MinerU path of the full-text chain: URL suffix and cleaning of its Markdown."""

from __future__ import annotations

from EvoScientist.memory.experiences import fulltext as ft

RAW = """# Title

## 1 Intro

Plain paragraph with E<sup>2</sup> and H<sub>2</sub>O.

![](images/abc.jpg)

Figure 1: a caption that is real text.

<table><tr><td>Model</td><td>F1</td></tr><tr><td>ALBERT</td><td>76.4</td></tr></table>



## References

[1] Someone. A paper.
"""


def test_clean_drops_figures_flattens_tables_keeps_text():
    out = ft.clean_mineru_markdown(RAW)
    assert "![" not in out and "<table" not in out and "<sup" not in out
    assert "Figure 1: a caption that is real text." in out
    assert "E2 and H2O" in out
    assert "Model | F1" in out and "ALBERT | 76.4" in out
    assert "\n\n\n" not in out


def test_mineru_url_gets_pdf_suffix_only_when_missing():
    assert ft._mineru_pdf_url("https://arxiv.org/pdf/2101.06326") == "https://arxiv.org/pdf/2101.06326.pdf"
    assert ft._mineru_pdf_url("https://arxiv.org/pdf/2101.06326v2.pdf") == "https://arxiv.org/pdf/2101.06326v2.pdf"
    assert ft._mineru_pdf_url("https://example.org/a.pdf?x=1") == "https://example.org/a.pdf?x=1"
