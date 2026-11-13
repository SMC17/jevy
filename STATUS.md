# jevy status

| | |
|---|---|
| Proof level | **unit-tested** (see [proof levels](https://github.com/SMC17/SMC17/blob/main/STANDARDS.md#proof-levels)) |
| Last verified | 2026-11-12, fresh clone, macOS on Apple M2 Max |
| Command | `cd zig && zig build test --summary all`; Python: README "Reproduce the dependency lock", then `pytest -q` |
| Result | Zig 153/153; Python 3.12 in a clean venv from `requirements.lock`: 194 passed, 12 skipped |

What is not claimed, and what comes next: [ROADMAP.md](ROADMAP.md) and the [open issues](https://github.com/SMC17/jevy/issues).
