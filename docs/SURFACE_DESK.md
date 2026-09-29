# Surface desk

**Version:** `1.0.0-zig-desk`  
**Code:** `jev_omm/surface/book.py`, `jev_omm/desk/fixtures.py`, existing `jev_omm/surface/svi.py` / `zig/src/svi.zig`.  
**Mode:** simulation / paper. The checked-in book is a synthetic fixture.

Raw SVI is still the fit. This layer is the book around it: one underlier id, a map of expiry to parameters, explicit spot / rate / dividend, butterfly and calendar gates, and an `iv_mkt − iv_fit` residual field.

Cite Gatheral & Jacquier, [Arbitrage-free SVI volatility surfaces](https://arxiv.org/abs/1204.0646).

## What a book holds

`SurfaceBook` is a dict of `UnderlierSurface`.

| Field | Meaning |
| --- | --- |
| `underlier_id` | `EQ_INDEX`, `EQ_SINGLE`, `FX_PAIR` in the fixture |
| `asset_class` | `equity_index`, `equity_single`, `fx_pair` |
| `beta_to_index` | 1.00, 1.35, 0.15 on those three names |
| `spot` | Input. `None` or non-positive stays missing |
| `rate`, `div_yield` | Used only to build the forward when spot is present |
| `slices` | Expiry in years → `SliceFit` |

A missing spot does not get a stand-in price. `forward()` returns `None`, `fit_quotes` stores an empty slice, and this book does not call the GEX converter. The existing GEX path already refuses a missing spot; the surface book does not invent one to get around that.

## Construction

```
quotes → total variance w = σ² T → raw SVI (Nelder–Mead, existing calibrate)
      → butterfly gate (density g, Lee slope)
      → calendar gate against the previous expiry
      → sticky-strike or sticky-delta mark
      → residual = iv_mkt − iv_fit
```

Expiries are fit short-dated first so the calendar check has a predecessor. `raw_calendar_ok` is the existing total-variance order on the k-grid. If a longer expiry prints below the shorter one, the slice is kept, `calendar_ok` is false, `surface_suspect` is true, and the **marked** total variance is lifted to the previous expiry pointwise (`damped=True`). The sleeve does not quote through the calendar break. The raw parameters stay on the slice so the failure is visible.

Sticky strike keeps log-moneyness against the fit forward. Sticky delta rebuilds k against the new forward. Both call the existing `implied_vol_after_move`.

`wing_residual` is the call-wing residual minus the put-wing residual, in vol points. `fly_residual` is the wing average minus the ATM residual. A smooth skew bump can sit inside the five SVI parameters, so those summaries can be near zero even when the quotes moved. A localized bump on a nine-strike grid leaves a point residual. The harness plants skew and fly states in the quotes and trades those states. The fitted field is the audit, not a second hidden alpha.

## Fixture

`jev_omm/data/fixtures/surfaces_synthetic.csv`

Every row has `synthetic_fixture=1`. Three names, three expiries (30/365, 90/365, 180/365), nine log-moneyness points. Smiles differ on purpose: index skew, a steeper single-name skew, a near-symmetric FX smile. Betas to the equity factor are 1.00, 1.35, and 0.15. Regenerating the file is `render_csv()` in `jev_omm/desk/fixtures.py`. Nothing in that file is an OPRA print.

## How sleeves consume it

The desk refits each name every `fit_stride` steps (default 20) inside `run_desk`. Suspect slices are counted. On the default seed, 10 of 36 slice fits came back `surface_suspect`. The book still marks them, with the calendar damp when that gate failed. Sleeves do not receive a live order from a suspect flag. The offline desk battery can raise `surface_suspect` when `desk.enabled` is on and the fit rmse is at least 0.02. With the gate off, that answer is not asked.
