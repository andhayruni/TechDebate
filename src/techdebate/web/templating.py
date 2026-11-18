"""The Jinja templates and their filters: `markdown` for the text of posts and comments, and `time` for when they were
written."""

from datetime import UTC, datetime, tzinfo

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from markupsafe import Markup
from starlette.templating import Jinja2Templates

from techdebate.web.markdown import render_markdown


def time_tag(moment: datetime, tz: tzinfo | None = None) -> Markup:
    """When something was written, in local time (`18 Nov 2026, 14:05`), with the exact UTC time for machines and in
    the tooltip. Local is the machine's time zone, which is the reader's, since the app runs on their machine."""
    local = moment.astimezone(tz)
    utc = moment.astimezone(UTC)
    return Markup('<time datetime="{}" title="{}">{}</time>').format(
        f"{utc:%Y-%m-%dT%H:%M:%SZ}", f"{utc:%Y-%m-%d %H:%M} UTC", f"{local.day} {local:%b %Y, %H:%M}"
    )


templates = Jinja2Templates(
    env=Environment(
        loader=PackageLoader("techdebate.web", "templates"),
        # Text from users and models is escaped, so it cannot add HTML to a page.
        autoescape=select_autoescape(),
        # A missing variable fails, as in the prompts, rather than leaving a gap in the page.
        undefined=StrictUndefined,
    )
)
templates.env.filters["markdown"] = render_markdown
templates.env.filters["time"] = time_tag
