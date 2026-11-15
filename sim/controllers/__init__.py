"""Shared command contract; each controller owns its internal state and diagnostics."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Command:
    """One interval's joint torque [Nm] or actuator position target [rad]."""

    kind: Literal["joint_torque", "joint_position"]
    values: np.ndarray
    indices: np.ndarray

    def apply(self, data):
        if self.kind == "joint_torque":
            data.qfrc_applied[self.indices] = self.values
        elif self.kind == "joint_position":
            data.ctrl[self.indices] = self.values
        else:
            raise ValueError(f"Unsupported command type: {self.kind}")
