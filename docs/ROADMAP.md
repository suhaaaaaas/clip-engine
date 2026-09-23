# Roadmap

Phases 1 and 2 run in parallel, one owner each, with a weekly sync where we
demo to each other and swap reviews. Everything runs on laptops; no cloud spend
until a creator says yes.

| Phase | Owner | Ships | Exit when |
| --- | --- | --- | --- |
| 0. Golden set | Both | Labeled clips for 3 episodes | Both label sets compared and merged |
| 1. Scoring | Suhaas | `clip score` + `clip eval` | 5+ of the 10 merged picks in its top 10, on the held-out episode |
| 2. Framing | Partner | `clip frame` + `clip render` | A side-by-side demo video either of us would show anyone |
| 3. Creator test | Both | 10 clips sent to one host | A clear yes, no, or "yes if…" |

Each phase ends with a short public write-up by its owner.

## After a yes

S3 upload trigger, a spot GPU worker, Postgres for clip state, a web review UI,
then posting to the creator's own accounts (YouTube first, TikTok after the app
audit). Roughly $30-35/month of cloud per daily show.

## Ground rules

- Downloaded episodes stay local. Clips go only to the host who owns them.
- $0 spend until Phase 3 gets a yes.
- Each of us checks our employer's outside-activity policy before this earns money.
- Ownership and roles in writing before any money shows up.
