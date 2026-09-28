"""Checked-in tape report: public-sample aggregates plus the synthetic fixture.

The OPRA section is frozen from one download of the no-account preview.
Raw rows are not in git. Re-fetch with ``scripts/fetch_databento_sample.py``
and compare the sha256 before treating the table as current.
"""

from __future__ import annotations

from pathlib import Path

from jev_omm.research.tape import TapeFeeSchedule, load_local
from jev_omm.research.tape_walk import render_walk_section, walk_forward

FIXTURE = Path("jev_omm/data/fixtures/tape_synthetic.csv")

# sha256 of the JSON body from GET /v0/dataset/sample on 2026-09-28.
# Aggregates only. The body itself is gitignored under data/local/.
CBBO_SHA256 = "b8ed8988cd064a35252049a6306f39a39040f54c1c1c8ebee1536a8a96882848"
TCBBO_SHA256 = "c1e45b59cc08a78041e800cec512fa8eb59ff8ab97f7f261c9f462d2e5f4cc52"
TCBBO_COUNTS = {
    "two_sided": 16,
    "crossed": 0,
    "locked": 3,
    "stale": 0,
    "one_sided": 0,
    "missing": 1,
}


def render_real_or_fixture_markdown() -> str:
    fixture = load_local(FIXTURE)
    # Same research fee as the frozen OPRA block so the two tables compare.
    walked = walk_forward(fixture, fees=TapeFeeSchedule(fee_per_contract=0.05, rebate_per_contract=0.0))
    counts = TCBBO_COUNTS
    intro = (
        "# Tape walk-forward (public preview and synthetic fixture)\n\n"
        "Research laboratory output. Two tapes, named for what they are.\n\n"
        "## What was loaded\n\n"
        "Databento's no-account sample endpoint "
        "`GET https://api.databento.com/v0/dataset/sample` returned a JSON "
        "list of CSV lines for `dataset=OPRA.PILLAR`. No API key. "
        "`schema=cbbo-1s` was an empty list on 2026-09-28. `schema=cbbo-1m` "
        "returned a header plus 20 rows: consolidated BBO and last trade for "
        "`TSLA  230901C00250000` (OSI call, expiry 2023-09-01, strike 250) "
        "from 2023-08-28 13:31Z through 13:50Z. That is a vendor preview, "
        "not a session and not a licensed history.\n\n"
        f"sha256 of the cbbo-1m JSON body: `{CBBO_SHA256}`.\n\n"
        "The underlying spot is not in the preview. It is left missing. "
        "No TSLA stock print was joined in from another sample (those files "
        "are a different product and, for the equity previews checked the "
        "same day, a different date).\n\n"
        "Redistribution of even a short OPRA preview is not clearly granted, "
        "so the raw body stays in gitignored `data/local/`. The table is "
        "aggregates from a local replay. Re-download with "
        "`python scripts/fetch_databento_sample.py` and check the sha256 "
        "before quoting the numbers as current.\n\n"
        "A second preview, `schema=tcbbo` (trade plus NBBO at the trade), "
        f"sha256 `{TCBBO_SHA256}`, has {counts['locked']} locked rows, "
        f"{counts['crossed']} crossed, {counts['missing']} missing bid/ask, "
        f"and {counts['two_sided']} two-sided, out of 20. Those locked prints "
        "are real preview rows. They are not mixed into the cbbo-1m replay "
        "below; the minute bars in cbbo-1m were two-sided.\n\n"
        "Posting rule: on a two-sided book, join the touch when the model is "
        "tighter and sit behind when it is wider. Never cross. The next row's "
        "trade fills us if it prints at or through our price. There is no "
        "queue ahead, so touch fills are overstated. Crossed, locked, stale, "
        "and missing books are not quoted. Fee 0.05 per contract, rebate 0, "
        "is a research schedule, not an OPRA fee card.\n\n"
        "The checked-in fixture `jev_omm/data/fixtures/tape_synthetic.csv` "
        "is `synthetic_fixture=1` on every row. The symbol `SYNTH` and the "
        "prices are invented. It is the runnable offline path. It is not OPRA.\n\n"
    )
    from jev_omm.research.frozen_opra import SECTION

    fixture_section = render_walk_section(walked, title="Synthetic fixture (invented prices)")
    return intro + SECTION + fixture_section


if __name__ == "__main__":
    print(render_real_or_fixture_markdown())
