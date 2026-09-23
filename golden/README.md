# Golden set

Hand-picked clips: the labels every model decision is measured against. Build
this before writing any scoring code.

## How to label

Both people label the same episode **separately**, then compare. Disagreement
is the interesting part: if the two of us only agree on 4 of 10 picks, a scorer
that hits 5 is doing well. Measure that overlap first; it sets the target.

## Format

One file per person per episode: `golden/<episode_id>_<name>.csv`

```csv
start_s,end_s,reason,type
412.5,448.0,"host calls the take indefensible, other two pile on",argument
1893.0,1921.5,"one-liner that lands, both laugh",joke
```

| Column | Meaning |
| --- | --- |
| `start_s` | Clip start in seconds from episode start |
| `end_s` | Clip end; aim for 20-60s |
| `reason` | One line, in your words, on why this would travel |
| `type` | One of: hot_take, big_moment, argument, joke, prediction |

## Rules

- 10 clips per episode per person. Stop at 10; forcing an eleventh teaches the
  model that mediocre moments are good.
- Label episodes 1 and 2 only. **Episode 3 is held out** -- never tune against it.
- Merged labels go in `golden/<episode_id>_merged.csv` after you compare.
