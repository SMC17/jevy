"""Multi-product paper desk: surfaces, sleeves, residual PnL, allocator.

Simulation only. TypeSafe / Jev answers are Choice / Score / Noul. They do
not emit orders. Live ``TYPESAFE_API_KEY`` is not required.
"""

from jev_omm.desk.allocator import allocate
from jev_omm.desk.fixtures import UNDERLIERS
from jev_omm.desk.harness import DeskConfig, run_desk
from jev_omm.desk.sleeves import SLEEVE_IDS, quote_or_target

__all__ = [
    "DeskConfig",
    "SLEEVE_IDS",
    "UNDERLIERS",
    "allocate",
    "quote_or_target",
    "run_desk",
]
