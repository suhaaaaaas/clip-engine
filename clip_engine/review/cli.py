"""Minimal review loop: play a clip, approve, reject, or adjust.

A terminal loop is enough to start. The web UI (player, editable transcript,
trim handles, framing override) only gets built once someone is reviewing
clips every week.

TODO(phase-3):
  - [ ] list pending clips for an episode
  - [ ] open a clip, take a keystroke: approve / reject / trim / re-frame
  - [ ] write decisions back to the EditList; never edit rendered video
  - [ ] re-scoring must not un-reject a clip a human already judged
"""


def review_episode(episode_id: str) -> None:
    raise NotImplementedError("TODO(phase-3)")
