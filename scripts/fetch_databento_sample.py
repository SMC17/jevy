#!/usr/bin/env python3
"""Download Databento's no-account sample previews into data/local/.

Does not use an API key. Does not open a live gateway. Writes raw JSON
the repo gitignores. Prints sha256 and book-label counts.

    python scripts/fetch_databento_sample.py
    python scripts/fetch_databento_sample.py --schema tcbbo
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_omm.research.databento_hist import fetch_public_sample  # noqa: E402
from jev_omm.research.tape import count_states  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="OPRA.PILLAR")
    parser.add_argument("--schema", default="cbbo-1m")
    parser.add_argument("--dest", default="data/local")
    args = parser.parse_args()
    path, tape, raw = fetch_public_sample(
        dataset=args.dataset,
        schema=args.schema,
        dest_dir=args.dest,
    )
    digest = hashlib.sha256(raw).hexdigest()
    print(f"wrote {path}")
    print(f"sha256 {digest}")
    print(f"rows {len(tape)} synthetic {tape.synthetic_fixture}")
    print(f"symbols {sorted(set(tape.symbol))}")
    print(f"book {count_states(tape)}")
    if tape.expiry:
        print(f"expiry {sorted(set(tape.expiry))} right {sorted(set(tape.right))}")


if __name__ == "__main__":
    main()
