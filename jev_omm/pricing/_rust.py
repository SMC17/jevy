"""Deprecated stub from abandoned Rust spike — use `_native` (Zig) instead."""

from jev_omm.pricing._native import (  # noqa: F401
    ZIG_AVAILABLE as RUST_AVAILABLE,  # misnomer retained for any old imports
    greeks,
    price,
    price_and_greeks,
)

__all__ = ["RUST_AVAILABLE", "price", "greeks", "price_and_greeks"]
