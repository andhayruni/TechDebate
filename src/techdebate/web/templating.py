"""The Jinja templates and their filters: `markdown` for the text of posts and comments, `time` for when they were
written, and `time_s`, with seconds, for the trace."""

from collections.abc import Callable
from datetime import UTC, datetime, tzinfo

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from markupsafe import Markup
from starlette.templating import Jinja2Templates

from techdebate.web.markdown import render_markdown


def time_tag(moment: datetime, tz: tzinfo | None = None) -> Markup:
    """When something was written, in local time (`18 Nov 2026, 14:05`), with the exact UTC time for machines and in
    the tooltip. Local is the machine's time zone, which is the reader's, since the app runs on their machine."""
    return _time(moment, tz, seconds=False)


def time_s_tag(moment: datetime, tz: tzinfo | None = None) -> Markup:
    """As `time_tag`, with seconds (`18 Nov 2026, 14:05:32`), for the trace, where the agents' visits overlap."""
    return _time(moment, tz, seconds=True)


def _time(moment: datetime, tz: tzinfo | None, *, seconds: bool) -> Markup:
    clock = "%H:%M:%S" if seconds else "%H:%M"
    local = moment.astimezone(tz)
    utc = moment.astimezone(UTC)
    return Markup('<time datetime="{}" title="{}">{}</time>').format(
        f"{utc:%Y-%m-%dT%H:%M:%SZ}", f"{utc:%Y-%m-%d {clock}} UTC", f"{local.day} {local:%b %Y, {clock}}"
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
templates.env.filters["time_s"] = time_s_tag


def macro(template: str, name: str) -> Callable[..., Markup]:
    """A macro of a template, called from Python, so what it renders there is what the page renders."""
    found: Callable[..., Markup] = getattr(templates.env.get_template(template).module, name)
    return found
