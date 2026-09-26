"""Pricing engines (BS first; local-vol / Heston upgrade path later).

Default path prefers the Zig shared library (`libjev_omm.so`) when present;
falls back to pure-Python `black_scholes.py`.
"""

from jev_omm.pricing._native import (
    ZIG_AVAILABLE,
    greeks,
    native_version,
    price,
    price_and_greeks,
)

__all__ = ["ZIG_AVAILABLE", "greeks", "native_version", "price", "price_and_greeks"]
