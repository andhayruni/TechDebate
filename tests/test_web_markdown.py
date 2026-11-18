import pytest

from techdebate.web.markdown import plain_text, render_markdown

LINK_ATTRS = 'target="_blank" rel="nofollow noopener noreferrer"'


def test_markdown_is_rendered() -> None:
    html = render_markdown("Rust is **growing**.\n\n- one\n- `two`")

    assert "<strong>growing</strong>" in html
    assert "<li><code>two</code></li>" in html


def test_raw_html_is_shown_as_text() -> None:
    html = render_markdown('<script>alert(1)</script> <img src="https://e.com/x.png" onerror="alert(2)">')

    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_an_http_link_opens_in_a_new_tab() -> None:
    html = render_markdown("[the repository](https://github.com/rust-lang/rust)")

    assert html.strip() == f'<p><a href="https://github.com/rust-lang/rust" {LINK_ATTRS}>the repository</a></p>'


@pytest.mark.parametrize(
    "link",
    [
        "[x](javascript:alert(1))",
        "[x](JavaScript:alert(1))",
        "[x](data:text/html,hi)",
        "[x](vbscript:msgbox)",
        "[x](file:///etc/passwd)",
        "[x](/posts/1)",
        "[x](#c3)",
        "<javascript:alert(1)>",
    ],
)
def test_only_http_links_are_made_links(link: str) -> None:
    html = render_markdown(link)

    assert "<a " not in html
    assert "href" not in html


def test_a_bare_url_becomes_a_link() -> None:
    html = render_markdown("See https://pypi.org/project/httpx/ for the numbers.")

    assert f'<a href="https://pypi.org/project/httpx/" {LINK_ATTRS}>https://pypi.org/project/httpx/</a>' in html


@pytest.mark.parametrize("text", ["Run setup.py and read README.md", "Mail me@example.com", "Try www.example.com"])
def test_a_name_without_a_scheme_is_not_a_link(text: str) -> None:
    """Without the scheme, file names would be linked: .py and .md are country domains."""
    assert "<a " not in render_markdown(text)


def test_an_image_is_shown_as_a_link_and_not_loaded() -> None:
    html = render_markdown("![a chart](https://example.com/chart.png) ![](https://example.com/x.png)")

    assert "<img" not in html
    assert f'<a href="https://example.com/chart.png" {LINK_ATTRS}>a chart</a>' in html
    assert f'<a href="https://example.com/x.png" {LINK_ATTRS}>https://example.com/x.png</a>' in html


def test_an_image_from_elsewhere_than_http_is_not_a_link() -> None:
    html = render_markdown("![x](javascript:alert(1))")

    assert "<a " not in html
    assert "<img" not in html


def test_plain_text_takes_out_the_markdown() -> None:
    text = "# Rust\n\nIs it **growing**? See [the survey](https://example.com/s) and `cargo`.\n\n- one\n- ![chart](https://e.com/c.png)"

    assert plain_text(text) == "Rust Is it growing? See the survey and cargo. one chart"


def test_plain_text_keeps_html_as_text() -> None:
    assert plain_text("<b>bold</b> claim") == "<b>bold</b> claim"
