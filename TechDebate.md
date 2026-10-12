# TechDebate: AI Panel for Assessing Technologies

Oct 6, 2026 · @Andok Hayruni

## The idea

A team of AI analysts that debates whether a technology is promising and will be widely used. A user names any technology, such as a new battery type, the Python language or AI in general, and each analyst decides on its own whether the topic is relevant and joins in. Each analyst gathers public evidence from a different angle and cites its sources.

Several AI analysts can take part. A few examples:

| Analyst | Question it asks | Sources |
| --- | --- | --- |
| Scientist | Is the idea sound, and are researchers still publishing on it? | arXiv, Semantic Scholar, OpenAlex |
| Builder | Are people actually using it? | GitHub, package downloads, Hacker News, Stack Overflow |
| Investor | Is money flowing in? | SEC filings, startup directories, press releases, job postings |
| Skeptic | Is it hype, or is interest fading? | Everyone's evidence, plus Google Trends, Wikipedia pageviews and critical articles |
| Historian | Which earlier technologies does it resemble, and how did they turn out? | Wikipedia timelines, past usage and research curves, a library of past technologies |
| Field expert | What does someone deep in this field make of it? | Its own library of papers and documentation, for example computer vision |

## How it works

1. **Choose a topic.** The user names a technology, and each analyst decides whether it is relevant and joins voluntarily.
2. **Gather evidence.** Each analyst collects public facts: research, usage, search interest and news.
3. **Debate.** Analysts challenge each other's arguments and evidence.
4. **Check claims.** Analysts can say whatever they want, but every figure and claim should trace to a source; others may criticise any claim with a missing citation.

Each analyst's relevance check uses a cheap model, so full effort goes only to analysts that join, and a spending limit caps the cost.

## Base version

**Functionality**

- Personal assistant: one user names the topic, with room to add more roles
- One field expert with its own knowledge base
- Evidence from free public sources
- A debate page showing the arguments, evidence and sources

**Deliverables**

- A live demo anyone can try in under a minute
- A public repository with a README on the design decisions
- A gallery of past discussions: a page where visitors can browse some of the previous discussions and open any of them to see the arguments, the evidence and the sources, without running anything

**What gets built**

| What you build | What it is here |
| --- | --- |
| Python service | Async Python, typed models, linter, `pytest` |
| LLM calls | One provider, with a cheap model for the relevance check and a stronger one for the debate |
| Agents and debate loop | Analyst agents plus a small orchestrator |
| Structured output | Claims as records: statement, number, source link, date |
| Data tools and MCP server | One function per source (GitHub, PyPI, arXiv, Hacker News, Google Trends), wrapped in one small MCP server |
| RAG (small) | Search over small libraries: past technologies for the Historian, and a field's papers and documentation for the field expert |
| Prompts and claim checker | Versioned prompt files, code that flags claims with missing citations, an optional AI reviewer |
| Database and cost cap | SQLite for debates, evidence and costs, plus a per-run and daily budget |
| Evals and tests | Fake-model tests that run offline |
| Tracing | Langfuse or OpenTelemetry, with every run viewable |
| Web page | FastAPI backend with a streamed debate view |
| Docker, CI/CD, deploy | Dockerfile, GitHub Actions for lint, tests and evals, and a simple host |

## Future work

- **Multiple users:** anyone can post a topic and join the discussion
- **A scout:** suggests topics worth debating automatically from news, research and developer activity
- **A verdict agent:** any user can ask it to summarize the discussion so far and draw a conclusion
- **More roles and data sources:** for example experts for other fields, patents, funding databases and industry reports
- **Accuracy tracking:** score verdicts against a test set of past technologies with known outcomes, and revisit old verdicts months later
- **Memory across debates:** track how a technology's outlook changes over time
- **Community feedback:** users rate the analysts' arguments, so strong ones rise to the top
- **Shareable debates:** a link to any finished discussion

**How the build grows**

| What changes | What it becomes later |
| --- | --- |
| Agents | More analyst roles, a verdict agent, and a scout that suggests topics |
| Data tools and MCP server | More sources such as patents, funding databases and industry reports, and authentication on the MCP server |
| RAG | Larger libraries, one per field, and a proper vector database |
| Event stream | Kafka feeding news items to the scout and the analysts |
| Database | Postgres, user accounts, per-user budgets and memory across debates |
| Feedback | Users rate the analysts' arguments, so strong ones rise |
| Evals | A test set of past technologies with known outcomes, and accuracy tracking that revisits old verdicts months later |
| LLM calls | Several providers behind a router, with fallbacks |
| Web page | Anyone can post a topic, join a discussion and share debates |
| Deployment | Managed cloud hosting and automated rollouts |
