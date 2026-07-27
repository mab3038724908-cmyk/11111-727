"""Pure contracts for TD25A operating and cleaning modes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple


class OperatingMode(str, Enum):
    IDLE = "IDLE"
    MANUAL_MAPPING = "MANUAL_MAPPING"
    AUTO_MAPPING = "AUTO_MAPPING"
    NAVIGATION = "NAVIGATION"
    CLEANING = "CLEANING"
    DOCKING = "DOCKING"
    MAINTENANCE = "MAINTENANCE"
    FAULT = "FAULT"


class CleaningPhase(str, Enum):
    INACTIVE = "INACTIVE"
    TRANSIT_UP = "TRANSIT_UP"
    FILL_DOWN = "FILL_DOWN"
    EDGE_RIGHT = "EDGE_RIGHT"
    EDGE_LEFT_FALLBACK = "EDGE_LEFT_FALLBACK"
    TURN_RETRACT = "TURN_RETRACT"
    TRANSFER_UP = "TRANSFER_UP"
    FINISHING = "FINISHING"
    FAULT_STOP = "FAULT_STOP"


class SafetyOverlay(str, Enum):
    NORMAL = "NORMAL"
    ULTRASONIC_ARMING = "ULTRASONIC_ARMING"
    ULTRASONIC_STOP = "ULTRASONIC_STOP"
    ESCAPING = "ESCAPING"
    HARDWARE_ESTOP = "HARDWARE_ESTOP"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"


class VerticalIntent(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    HOLD = "HOLD"


@dataclass(frozen=True)
class CleaningIntent:
    vertical: VerticalIntent
    x_extension_m: float
    cleaning_enabled: bool


@dataclass(frozen=True)
class EdgeUltrasonicSelection:
    fixed_mid_channel: int
    fixed_rear_channel: int
    moving_guard_channel: int
    forward_channels: Tuple[int, ...]

    @property
    def poll_channels(self) -> Tuple[int, ...]:
        return tuple(sorted({
            self.fixed_mid_channel,
            self.fixed_rear_channel,
            *self.forward_channels,
        }))

    @property
    def poll_mask(self) -> int:
        mask = 0
        for channel in self.poll_channels:
            mask |= 1 << (channel - 1)
        return mask


RIGHT_EDGE_ULTRASONICS = EdgeUltrasonicSelection(
    fixed_mid_channel=10,
    fixed_rear_channel=8,
    moving_guard_channel=9,
    forward_channels=(4, 6, 7),
)

LEFT_EDGE_ULTRASONICS = EdgeUltrasonicSelection(
    fixed_mid_channel=2,
    fixed_rear_channel=5,
    moving_guard_channel=3,
    forward_channels=(4, 6, 7),
)


CLEAN_RIGHT_EDGE_AT_X0_M = 0.5548
X_MAX_EXTENSION_M = 0.250
FIXED_SIDE_SENSOR_OFFSET_M = 0.270
FIXED_SIDE_SENSOR_BASELINE_M = 0.429
DEFAULT_EDGE_GAP_M = 0.030


@dataclass(frozen=True)
class EdgeExtension:
    wall_distance_m: float
    requested_extension_m: float
    extension_m: float
    expected_cleaning_gap_m: float
    reachable: bool


def _valid_ranges(values: Iterable[Optional[float]]) -> Tuple[float, ...]:
    return tuple(
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def wall_distance_from_fixed_side_ranges(
    mid_range_m: Optional[float],
    rear_range_m: Optional[float],
    sensor_offset_m: float = FIXED_SIDE_SENSOR_OFFSET_M,
) -> float:
    """Estimate base-link centre-line distance to a parallel side wall."""
    ranges = _valid_ranges((mid_range_m, rear_range_m))
    if not ranges:
        raise ValueError("at least one valid fixed side range is required")
    return float(sensor_offset_m) + sum(ranges) / len(ranges)


def wall_alignment_error_rad(
    mid_range_m: float,
    rear_range_m: float,
    baseline_m: float = FIXED_SIDE_SENSOR_BASELINE_M,
) -> float:
    """Return the small wall/robot yaw error inferred by the fixed pair."""
    if not all(math.isfinite(value) for value in (
            mid_range_m, rear_range_m, baseline_m)):
        raise ValueError("wall alignment inputs must be finite")
    if mid_range_m <= 0.0 or rear_range_m <= 0.0 or baseline_m <= 0.0:
        raise ValueError("wall alignment inputs must be positive")
    return math.atan2(mid_range_m - rear_range_m, baseline_m)


def edge_extension_for_wall_distance(
    wall_distance_m: float,
    desired_gap_m: float = DEFAULT_EDGE_GAP_M,
    clean_edge_at_x0_m: float = CLEAN_RIGHT_EDGE_AT_X0_M,
    max_extension_m: float = X_MAX_EXTENSION_M,
) -> EdgeExtension:
    """Compute a fixed X setpoint for one straight right-edge segment."""
    values = (
        wall_distance_m, desired_gap_m, clean_edge_at_x0_m, max_extension_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("edge geometry must be finite")
    if wall_distance_m <= 0.0:
        raise ValueError("wall distance must be positive")
    if desired_gap_m < 0.0 or clean_edge_at_x0_m <= 0.0:
        raise ValueError("edge geometry is invalid")
    if max_extension_m < 0.0:
        raise ValueError("maximum extension must be non-negative")

    requested = wall_distance_m - clean_edge_at_x0_m - desired_gap_m
    extension = min(max_extension_m, max(0.0, requested))
    expected_gap = wall_distance_m - clean_edge_at_x0_m - extension
    reachable = (
        -1e-9 <= requested <= max_extension_m + 1e-9
        and expected_gap <= desired_gap_m + 1e-9
    )
    return EdgeExtension(
        wall_distance_m=wall_distance_m,
        requested_extension_m=requested,
        extension_m=extension,
        expected_cleaning_gap_m=expected_gap,
        reachable=reachable,
    )


def edge_extension_from_fixed_ranges(
    mid_range_m: Optional[float],
    rear_range_m: Optional[float],
    desired_gap_m: float = DEFAULT_EDGE_GAP_M,
) -> EdgeExtension:
    wall_distance = wall_distance_from_fixed_side_ranges(
        mid_range_m, rear_range_m)
    return edge_extension_for_wall_distance(
        wall_distance, desired_gap_m=desired_gap_m)


def intent_for_cleaning_phase(
    phase: CleaningPhase,
    edge_extension_m: float = 0.0,
) -> CleaningIntent:
    """Map semantic path phases to fail-safe actuator intent."""
    extension = min(X_MAX_EXTENSION_M, max(0.0, float(edge_extension_m)))
    if phase == CleaningPhase.FILL_DOWN:
        return CleaningIntent(VerticalIntent.DOWN, 0.0, True)
    if phase == CleaningPhase.EDGE_RIGHT:
        return CleaningIntent(VerticalIntent.DOWN, extension, True)
    if phase == CleaningPhase.EDGE_LEFT_FALLBACK:
        return CleaningIntent(VerticalIntent.DOWN, 0.0, True)
    if phase in (
            CleaningPhase.TRANSIT_UP,
            CleaningPhase.TRANSFER_UP,
            CleaningPhase.FINISHING):
        return CleaningIntent(VerticalIntent.UP, 0.0, False)
    if phase in (
            CleaningPhase.TURN_RETRACT,
            CleaningPhase.FAULT_STOP,
            CleaningPhase.INACTIVE):
        return CleaningIntent(VerticalIntent.HOLD, 0.0, False)
    raise ValueError(f"unsupported cleaning phase: {phase}")
