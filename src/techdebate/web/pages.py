"""What the pages show, built from the store's records: the feed, and a post with its threads. The templates only lay
these out, so the shape of a thread is decided here, from `Tree`, and a single comment renders the same wherever it
appears."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from techdebate.errors import usd_limit
from techdebate.models import Message, User, UserKind
from techdebate.roles import display_name
from techdebate.spending import Spent, Today, duration
from techdebate.store import PostSummary
from techdebate.tree import Tree
from techdebate.views import cut
from techdebate.web.markdown import plain_text

# How much of a post the feed shows, and how much names a post page in the browser's tab.
FEED_START_CHARS = 200
TITLE_CHARS = 60


@dataclass(frozen=True)
class Author:
    """A name as it is shown. An agent's is always shown with the AI badge."""

    name: str
    is_agent: bool


def author_of(user: User) -> Author:
    if user.kind is UserKind.AGENT:
        return Author(name=display_name(user.username), is_agent=True)
    return Author(name=user.username, is_agent=False)


@dataclass(frozen=True)
class FeedItem:
    id: str
    author: Author
    created_at: datetime
    start: str  # the start of the text, as plain text
    comments: int


def feed(summaries: Sequence[PostSummary], users: Mapping[str, User]) -> list[FeedItem]:
    """The posts in the order given, newest first from the store. `users` maps each author's ID to its user."""
    return [
        FeedItem(
            id=summary.post.id,
            author=author_of(users[summary.post.author_id]),
            created_at=summary.post.created_at,
            start=_start(summary.post.text, FEED_START_CHARS),
            comments=summary.comments,
        )
        for summary in summaries
    ]


@dataclass(frozen=True)
class ReplyTo:
    """The comment a reply answers, when that is not the top-level comment of its thread."""

    number: int
    author: Author


@dataclass(frozen=True)
class CommentView:
    id: str
    number: int
    author: Author
    created_at: datetime
    text: str  # markdown, as written
    reply_to: ReplyTo | None


@dataclass(frozen=True)
class PostPage:
    id: str
    author: Author
    created_at: datetime
    text: str  # markdown, as written
    title: str  # the start of the text, as plain text
    comments: int
    threads: list[list[CommentView]]  # each thread is its top-level comment, then its replies, oldest first


def post_page(tree: Tree, users: Mapping[str, User]) -> PostPage:
    """The post and every thread, oldest first. Every reply in a thread is shown one level under its top-level
    comment; one that answers another reply names it, as `show` does (4.5)."""

    def view(comment: Message) -> CommentView:
        parent = tree.parent(comment)
        top = tree.top_level(comment)
        reply_to = None
        if parent.id not in (tree.post.id, top.id):
            reply_to = ReplyTo(number=tree.number(parent), author=author_of(users[parent.author_id]))
        return CommentView(
            id=comment.id,
            number=tree.number(comment),
            author=author_of(users[comment.author_id]),
            created_at=comment.created_at,
            text=comment.text,
            reply_to=reply_to,
        )

    post = tree.post
    return PostPage(
        id=post.id,
        author=author_of(users[post.author_id]),
        created_at=post.created_at,
        text=post.text,
        title=_start(post.text, TITLE_CHARS),
        comments=len(tree.comments),
        threads=[[view(comment) for comment in thread] for thread in tree.threads()],
    )


def _start(text: str, limit: int) -> str:
    """The start of the text as plain text, or of the text as written when its markdown leaves nothing, such as an
    image with no alt text."""
    return cut(plain_text(text) or " ".join(text.split()), limit)


# The budgets


@dataclass(frozen=True)
class Notice:
    """A budget that is used up, which the pages say under the header: the day's, an agent's or the post's."""

    kind: Literal["day", "agent", "post"]
    limit: str  # as usd_limit shows it
    resets_in: str | None  # for the daily budgets
    author: Author | None = None  # for an agent's


def notices(today: Today, post: Spent | None = None) -> list[Notice]:
    """The budgets used up: the day's, or else each agent's, and the post's. With the day used up every agent has
    stopped, so the agents' own are not repeated."""
    resets_in = duration(today.resets_in)
    found: list[Notice] = []
    if today.all_agents.used_up:
        found.append(Notice("day", usd_limit(today.all_agents.limit_usd), resets_in))
    else:
        for agent, spent in today.agents:
            if spent.used_up:
                found.append(Notice("agent", usd_limit(spent.limit_usd), resets_in, author_of(agent)))
    if post is not None and post.used_up:
        found.append(Notice("post", usd_limit(post.limit_usd), None))
    return found


@dataclass(frozen=True)
class BudgetRow:
    """One budget: what it has spent, against its limit, and what is left."""

    spent: str
    limit: str
    left: str  # "$1.7900 left", or "used up"
    used_up: bool


def budget_row(spent: Spent) -> BudgetRow:
    left = "used up" if spent.used_up else f"${spent.left_usd:.4f} left"
    return BudgetRow(f"${spent.spent_usd:.4f}", usd_limit(spent.limit_usd), left, spent.used_up)


@dataclass(frozen=True)
class PostBudget:
    id: str
    start: str
    row: BudgetRow


@dataclass(frozen=True)
class BudgetPage:
    """What `techdebate budget` shows, and every post's spending."""

    resets_in: str
    all_agents: BudgetRow
    agents: list[tuple[Author, BudgetRow]]
    posts: list[PostBudget]  # newest first


def budget_page(today: Today, posts: Sequence[tuple[Message, Spent]]) -> BudgetPage:
    return BudgetPage(
        resets_in=duration(today.resets_in),
        all_agents=budget_row(today.all_agents),
        agents=[(author_of(agent), budget_row(spent)) for agent, spent in today.agents],
        posts=[PostBudget(post.id, _start(post.text, TITLE_CHARS), budget_row(spent)) for post, spent in posts],
    )
