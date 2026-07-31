"""Grid-map coverage planner for indoor cleaning robots.

The planner consumes a Nav2-style OccupancyGrid snapshot and returns one
continuous polyline:

  robot pose -> connector -> swath -> connector -> swath -> ...

Swaths are generated directly on the inflated free-space grid, so cleaning
segments cannot cross occupied or unknown cells. Offline connector segments
prefer mask-constrained Theta* on the same free-space mask and retain A* as a
safe fallback.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from td25a_robot_ui.algorithms.free_space import (
    dilate_binary,
    erode_binary,
    largest_component,
    polygon_to_mask,
)

Point = Tuple[float, float]
Cell = Tuple[int, int]  # (row, col)
Swath = Tuple[Point, Point]


class BoundaryType(str, Enum):
    """Semantic source of a route-side boundary, independent of actuation."""

    NONE = "NONE"
    PHYSICAL_WALL = "PHYSICAL_WALL"
    PHYSICAL_OBSTACLE = "PHYSICAL_OBSTACLE"
    PARTITION_SEAM = "PARTITION_SEAM"
    SELECTION_BOUNDARY = "SELECTION_BOUNDARY"
    DOOR = "DOOR"
    OPEN_BOUNDARY = "OPEN_BOUNDARY"


class CleanerMode(str, Enum):
    """Operator-facing cleaning mode carried alongside every base pose."""

    EDGE_LEFT = "EDGE_LEFT"
    EDGE_RIGHT = "EDGE_RIGHT"
    EDGE_CENTER = "EDGE_CENTER"
    EDGE_RETRACT = "EDGE_RETRACT"
    EDGE_IGNORE = "EDGE_IGNORE"


@dataclass(frozen=True)
class CleanerCommand:
    """One simulated lateral-cleaner command aligned with a path point.

    Positive offset is body-left and negative offset is body-right.  The
    standard ROS Pose remains untouched; this metadata is consumed by a
    separate cleaner controller when real actuator integration is enabled.
    """

    mode: CleanerMode = CleanerMode.EDGE_CENTER
    offset_m: float = 0.0
    boundary_type: BoundaryType = BoundaryType.NONE
    cleaning_enabled: bool = True


def _clearance_pad_cells(clearance_m: float, resolution: float) -> int:
    """Convert centre clearance to occupied-cell dilation iterations.

    An occupied grid cell already has half a cell of physical extent.  After
    ``k`` dilations the nearest surviving centre is ``k+1`` cells away, so its
    boundary clearance is ``(k+0.5)*resolution``.  Accounting for that half cell
    prevents an 0.34m footprint from being rounded up to an artificial 0.40m on
    an 0.08m map and closing a physically passable doorway.
    """
    if resolution <= 0.0:
        raise ValueError("resolution must be > 0")
    return max(1, int(math.ceil(max(0.0, clearance_m) / resolution - 0.5)))


@dataclass
class CoveragePlan:
    path: List[Point]
    swaths: List[Swath]
    free_mask: np.ndarray
    snapped_start: Point
    perimeter_start_idx: int = -1   # path 中"沿边收尾段"起点下标; -1 = 无沿边段


@dataclass
class CoverageFootprint:
    """TD25A body geometry used by partitioned coverage planning.

    The narrow-door topology uses ``half_width_m`` only after the robot has
    aligned with the doorway.  Turns and room coverage use the circumscribed
    rear-corner radius, which is much larger for TD25A's asymmetric body.
    """

    front_m: float = 0.33
    rear_m: float = 0.72
    half_width_m: float = 0.34
    # Exact body sweep uses the configured polygon dimensions.  Room turns use
    # an additional radial reserve; adding that reserve to the door width too
    # would incorrectly declare the robot's known 0.68m-width passages closed.
    tracking_margin_m: float = 0.0
    turn_margin_m: float = 0.07

    @property
    def turn_clearance_m(self) -> float:
        return max(
            math.hypot(
                self.front_m + self.turn_margin_m,
                self.half_width_m + self.turn_margin_m,
            ),
            math.hypot(
                self.rear_m + self.turn_margin_m,
                self.half_width_m + self.turn_margin_m,
            ),
        )

    @property
    def polygon(self) -> List[Point]:
        return [
            (self.front_m, self.half_width_m),
            (self.front_m, -self.half_width_m),
            (-self.rear_m, -self.half_width_m),
            (-self.rear_m, self.half_width_m),
        ]


@dataclass
class CoverageRegion:
    region_id: int
    mask: np.ndarray              # turn-safe cleaning-centre mask
    bbox_cells: Tuple[int, int, int, int]
    centroid: Point
    area_m2: float
    axis: str
    cell_count: int = 1
    travel_mask: Optional[np.ndarray] = None
    fragmentation_profiles: List[Dict[str, float]] = field(
        default_factory=list)


@dataclass
class CoverageSegment:
    kind: str  # transfer | fill | perimeter
    region_id: int
    path: List[Point]
    from_region_id: int = -1
    to_region_id: int = -1
    swaths: List[Swath] = field(default_factory=list)
    path_start_idx: int = -1
    path_end_idx: int = -1
    # When true, the next segment is only a visual/semantic colour change; the
    # controller must keep following the tangent-continuous geometry without a
    # stop-and-rotate at this shared point.
    continuous_to_next: bool = False
    # Segment-level default plus an optional point-aligned profile.  Fill and
    # transfer segments remain centred in v1; perimeter segments carry the
    # classified wall/door/seam command at every point without changing their
    # already-certified base geometry.
    cleaner_mode: CleanerMode = CleanerMode.EDGE_CENTER
    cleaner_offset_m: float = 0.0
    cleaner_profile: List[CleanerCommand] = field(default_factory=list)
    # Diagnostics for partitioned fill segments.  Each tuple is
    # ``(source_component_id, first_swath_index, end_swath_index)`` with an
    # exclusive end.  Keeping the BCD ownership alongside the flattened swath
    # list lets offline review distinguish a U-turn inside one cell from the
    # connector used to move between cells; it does not alter execution.
    component_swath_ranges: List[Tuple[int, int, int]] = field(
        default_factory=list)
    component_transition_diagnostics: List[Dict[str, object]] = field(
        default_factory=list)


@dataclass
class PartitionedCoveragePlan:
    path: List[Point]
    segments: List[CoverageSegment]
    regions: List[CoverageRegion]
    visit_order: List[int]
    free_mask: np.ndarray
    snapped_start: Point
    swaths: List[Swath] = field(default_factory=list)
    # FollowPath must never see several adjacent 0.525 m lanes in one goal.
    # Stops are shared endpoints: the previous goal arrives with the incoming
    # yaw, then RotationShim aligns the next goal to the outgoing yaw.
    hard_stop_indices: List[int] = field(default_factory=list)
    failure_reason: str = ""
    footprint_valid: bool = False
    footprint_violation_count: int = 0
    coverage_complete: bool = False
    path_continuous: bool = False
    max_segment_gap_m: float = 0.0
    turn_safe_coverage_ratio: float = 0.0
    # Executable floor KPI: cells within brush reach of a centre pose where
    # the complete asymmetric body has turn clearance.
    serviceable_coverage_ratio: float = 0.0
    # Same serviceable target, but stamped with the real TD25A forward brush
    # rectangle and the explicit arrival/departure pose at every route point.
    # This is the acceptance KPI that can be compared directly with the live
    # CleanedAreaTracker counts after the one-shot route finishes.
    actual_brush_coverage_ratio: float = 0.0
    reachable_coverage_ratio: float = 0.0
    region_coverage_ratios: Dict[int, float] = field(default_factory=dict)
    region_serviceable_coverage_ratios: Dict[int, float] = field(
        default_factory=dict)
    region_actual_brush_coverage_ratios: Dict[int, float] = field(
        default_factory=dict)
    # A rectangular single-door room can be planned in two deterministic
    # ways: the proven legacy ordering, or an exit-aware ordering that tries
    # to finish near its doorway and avoids crossing completed lanes.  The
    # public planner keeps the legacy result unless the alternative passes all
    # safety/coverage invariants and strictly improves the route-quality gate.
    selection_mode: str = "baseline"
    exit_aware_candidate_region_ids: List[int] = field(default_factory=list)
    current_first_candidate_recommended: bool = False
    graph_order_candidate_recommended: bool = False
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    alternative_quality_metrics: Dict[str, float] = field(default_factory=dict)
    # A hard-stop point has two legitimate poses at the same (x, y): arrive
    # along the completed lane, then depart along the next connector/lane after
    # RotationShim aligns the robot.  Keeping both arrays prevents the UI from
    # recreating heading semantics from a sliced polyline later.
    arrival_yaws: List[float] = field(default_factory=list)
    departure_yaws: List[float] = field(default_factory=list)
    cleaner_profile: List[CleanerCommand] = field(default_factory=list)
    cleaner_center_path: List[Point] = field(default_factory=list)
    centered_brush_coverage_ratio: float = 0.0
    cleaner_extension_gain_area_m2: float = 0.0
    cleaner_semantics_valid: bool = False
    cleaner_semantics_failure_reason: str = ""
    cleaner_mode_point_counts: Dict[str, int] = field(default_factory=dict)
    boundary_type_point_counts: Dict[str, int] = field(default_factory=dict)
    cleaner_max_offset_m: float = 0.0
    # Execution-side current-pose lead-ins are not part of the offline nominal
    # route.  Preserve the exact masks/body used by the planner so those short
    # recoupling segments can be checked with the same collision contract
    # instead of assuming that a straight line between two free centres is
    # safe for TD25A's long asymmetric rear overhang.
    raw_free_mask: Optional[np.ndarray] = None
    serviceable_target_mask: Optional[np.ndarray] = None
    # Floor excluded from the acceptance denominator because every associated
    # turn-safe island is both smaller than the configured useful-area floor
    # and too short for a meaningful straight cleaning pass.  Keeping this
    # explicit makes the green review boundary and the >=95% KPI use the same
    # denominator instead of silently treating tiny lidar pockets as cleaned.
    discarded_small_component_count: int = 0
    discarded_small_area_m2: float = 0.0
    # Optional post-pass diagnostics.  A refined transfer is accepted only
    # when its proper crossings and centre-line overlap both improve without
    # adding turns, leaving all fill/perimeter geometry unchanged.
    refined_transfer_count: int = 0
    refined_transfer_crossing_reduction: int = 0
    refinement_diagnostics: Dict[str, int] = field(default_factory=dict)
    footprint: CoverageFootprint = field(default_factory=CoverageFootprint)


def build_coverage_free_mask(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    robot_radius_m: float,
    clip_polygon: Optional[Sequence[Point]] = None,
    blocked_polygons: Optional[Sequence[Sequence[Point]]] = None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> np.ndarray:
    """Return True where the robot center may safely travel."""
    grid = np.frombuffer(data, dtype=np.int8).reshape(height, width)
    obstacle = grid > 50
    unknown = grid < 0
    pad_cells = _clearance_pad_cells(robot_radius_m, resolution)
    not_walkable = dilate_binary(obstacle, pad_cells) | unknown
    free = ~not_walkable
    free[:pad_cells, :] = False
    free[-pad_cells:, :] = False
    free[:, :pad_cells] = False
    free[:, -pad_cells:] = False
    if clip_polygon is not None:
        free &= polygon_to_mask(
            clip_polygon, width, height, resolution, origin_x, origin_y)
    for blocked in blocked_polygons or ():
        free &= ~polygon_to_mask(
            blocked, width, height, resolution, origin_x, origin_y)
    return free


def _build_radial_clearance_mask(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    clearance_m: float,
    blocked_polygons: Optional[Sequence[Sequence[Point]]] = None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> np.ndarray:
    """Euclidean clearance mask for phases where the body may rotate.

    The legacy helper uses a 4-neighbour diamond dilation.  That is useful for
    preserving narrow-door topology, but a diamond does not contain the rear
    corner of a rotated 1.05m x 0.68m TD25A.  This disk dilation is deliberately
    local to the new planner so established online behaviour is unchanged.
    """
    grid = np.frombuffer(data, dtype=np.int8).reshape(height, width)
    blocked = (grid > 50) | (grid < 0)
    for polygon in blocked_polygons or ():
        blocked |= polygon_to_mask(
            polygon, width, height, resolution, origin_x, origin_y)
    # Occupied cells are squares.  Include their farthest half-cell corner so
    # surviving centres have the requested Euclidean boundary clearance.
    radius = max(0.0, float(clearance_m)) + resolution / math.sqrt(2.0)
    extent = max(1, int(math.ceil(radius / resolution)))
    dilated = np.zeros_like(blocked, dtype=bool)
    columns = np.arange(width, dtype=np.int64)
    for dy in range(-extent, extent + 1):
        remaining_sq = radius * radius - (dy * resolution) ** 2
        if remaining_sq < -1e-12:
            continue
        half_width = int(math.floor(
            math.sqrt(max(0.0, remaining_sq)) / resolution + 1e-12))
        y0_src = max(0, -dy)
        y1_src = min(height, height - dy)
        y0_dst = y0_src + dy
        y1_dst = y1_src + dy
        source = blocked[y0_src:y1_src]
        # Horizontal window OR via prefix sums: one full-row pass for this dy,
        # instead of one full-image boolean shift for every dx in the disk.
        prefix = np.pad(
            np.cumsum(source, axis=1, dtype=np.int32),
            ((0, 0), (1, 0)), mode="constant")
        left = np.maximum(0, columns - half_width)
        right = np.minimum(width, columns + half_width + 1)
        horizontal = (prefix[:, right] - prefix[:, left]) > 0
        dilated[y0_dst:y1_dst] |= horizontal
    free = ~dilated
    free[:extent, :] = False
    free[-extent:, :] = False
    free[:, :extent] = False
    free[:, -extent:] = False
    return free


def _disk_dilate_mask(
    mask: np.ndarray,
    resolution: float,
    radius_m: float,
) -> np.ndarray:
    """Dilate a boolean mask by a Euclidean disk for coverage accounting."""
    if not mask.any() or radius_m <= 0.0:
        return mask.copy()
    # A target grid cell counts as covered when the cleaning disk overlaps any
    # part of it, hence the half-cell-corner allowance.
    radius = float(radius_m) + resolution / math.sqrt(2.0)
    extent = max(1, int(math.ceil(radius / resolution)))
    height, width = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)
    for dy in range(-extent, extent + 1):
        remaining = radius * radius - (dy * resolution) ** 2
        if remaining < -1e-12:
            continue
        dx_limit = int(math.floor(
            math.sqrt(max(0.0, remaining)) / resolution + 1e-12))
        source_r0 = max(0, -dy)
        source_r1 = min(height, height - dy)
        target_r0 = source_r0 + dy
        target_r1 = source_r1 + dy
        for dx in range(-dx_limit, dx_limit + 1):
            source_c0 = max(0, -dx)
            source_c1 = min(width, width - dx)
            target_c0 = source_c0 + dx
            target_c1 = source_c1 + dx
            dilated[target_r0:target_r1, target_c0:target_c1] |= (
                mask[source_r0:source_r1, source_c0:source_c1])
    return dilated


def _polyline_cleaning_mask(
    shape: Tuple[int, int],
    path: Sequence[Point],
    resolution: float,
    origin_x: float,
    origin_y: float,
    clean_width_m: float,
) -> np.ndarray:
    """Rasterize a route and expand it by half the cleaning width."""
    return _polylines_cleaning_mask(
        shape, [path], resolution, origin_x, origin_y, clean_width_m)


def _polylines_cleaning_mask(
    shape: Tuple[int, int],
    paths: Sequence[Sequence[Point]],
    resolution: float,
    origin_x: float,
    origin_y: float,
    clean_width_m: float,
) -> np.ndarray:
    """Rasterize independent polylines, then dilate their union once."""
    centres = np.zeros(shape, dtype=bool)
    if not paths or clean_width_m <= 0.0:
        return centres
    height, width = shape
    sample_step = max(0.01, resolution * 0.40)
    for path in paths:
        if not path:
            continue
        pairs = (
            zip(path, path[1:])
            if len(path) >= 2 else ((path[0], path[0]),))
        for start, end in pairs:
            distance = _dist(start, end)
            sample_count = max(1, int(math.ceil(distance / sample_step)))
            for sample_index in range(sample_count + 1):
                ratio = sample_index / sample_count
                point = (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
                row, col = _world_to_cell(
                    point, resolution, origin_x, origin_y)
                if 0 <= row < height and 0 <= col < width:
                    centres[row, col] = True
    return _disk_dilate_mask(
        centres, resolution, clean_width_m * 0.5)


def _forward_brush_cleaning_mask(
    shape: Tuple[int, int],
    path: Sequence[Point],
    arrival_yaws: Sequence[float],
    departure_yaws: Sequence[float],
    resolution: float,
    origin_x: float,
    origin_y: float,
    cleaner_profile: Optional[Sequence[CleanerCommand]] = None,
) -> np.ndarray:
    """Stamp the exact live-tracker brush model along an explicit-pose route.

    A hard-stop coordinate is physically occupied twice: first with the lane
    arrival yaw, then after RotationShim's in-place alignment with the next
    departure yaw.  Stamping both poses is therefore part of the executable
    contract, not an optimistic interpolation.  Intermediate rotations only
    add swept brush area, so omitting them keeps this offline KPI conservative.
    """
    cleaned = np.zeros(shape, dtype=np.uint16)
    if not path:
        return cleaned.astype(bool)
    if (len(arrival_yaws) != len(path)
            or len(departure_yaws) != len(path)):
        raise ValueError("explicit brush yaw arrays must match path length")
    if cleaner_profile is not None and len(cleaner_profile) != len(path):
        raise ValueError("cleaner profile must match path length")

    # Import the single production implementation rather than duplicating its
    # measured +0.14..+0.52 m by +/-0.34 m geometry here.
    from td25a_robot_ui.store.cleaned_area import paint_brush

    for index, (point, arrival_yaw, departure_yaw) in enumerate(zip(
            path, arrival_yaws, departure_yaws)):
        command = (
            cleaner_profile[index]
            if cleaner_profile is not None else CleanerCommand())
        if not command.cleaning_enabled:
            continue
        paint_brush(
            cleaned, resolution, origin_x, origin_y,
            float(point[0]), float(point[1]), float(arrival_yaw),
            lateral_offset_m=float(command.offset_m))
        if abs(_wrap_angle(
                float(departure_yaw) - float(arrival_yaw))) > 1e-6:
            paint_brush(
                cleaned, resolution, origin_x, origin_y,
                float(point[0]), float(point[1]), float(departure_yaw),
                lateral_offset_m=float(command.offset_m))
    return cleaned > 0


def _brush_pose_is_free(
    raw_free_mask: np.ndarray,
    point: Point,
    yaw: float,
    lateral_offset_m: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> bool:
    """Check the measured forward brush rectangle at one lateral offset."""
    from td25a_robot_ui.store.cleaned_area import (
        BRUSH_LEFT_M,
        BRUSH_RIGHT_M,
        BRUSH_X0_M,
        BRUSH_X1_M,
    )

    step = max(0.015, resolution * 0.35)
    xs = np.arange(BRUSH_X0_M, BRUSH_X1_M + step * 0.5, step)
    ys = np.arange(-BRUSH_RIGHT_M, BRUSH_LEFT_M + step * 0.5, step)
    xx, yy = np.meshgrid(xs, ys + float(lateral_offset_m))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    wx = point[0] + cosine * xx.ravel() - sine * yy.ravel()
    wy = point[1] + sine * xx.ravel() + cosine * yy.ravel()
    cols = np.floor((wx - origin_x) / resolution).astype(np.int64)
    rows = np.floor((wy - origin_y) / resolution).astype(np.int64)
    valid = (
        (rows >= 0) & (rows < raw_free_mask.shape[0])
        & (cols >= 0) & (cols < raw_free_mask.shape[1]))
    return bool(valid.all() and raw_free_mask[rows, cols].all())


def _annotate_cleaner_semantics(
    segments: Sequence[CoverageSegment],
    global_path: Sequence[Point],
    regions: Sequence[CoverageRegion],
    raw_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    selection_boundary_mask: Optional[np.ndarray],
    resolution: float,
    origin_x: float,
    origin_y: float,
    footprint: CoverageFootprint,
    max_offset_m: float,
    wall_gap_m: float,
    transition_distance_m: float,
) -> Tuple[List[CleanerCommand], List[Point], bool, str,
           Dict[str, int], Dict[str, int]]:
    """Classify red path points and attach a simulated cleaner profile.

    This pass never changes the certified base route.  It distinguishes a real
    occupied wall from room seams, doors and manual selection edges, then moves
    only the measured brush frame.  Every proposed offset is reduced until its
    complete brush rectangle is collision-free on the saved static map.
    """
    max_offset = max(0.0, float(max_offset_m))
    transition_distance = max(0.10, float(transition_distance_m))
    from td25a_robot_ui.store.cleaned_area import BRUSH_LEFT_M, BRUSH_RIGHT_M

    selection_near = None
    if selection_boundary_mask is not None and selection_boundary_mask.any():
        selection_boundary = selection_boundary_mask & ~erode_binary(
            selection_boundary_mask, 1)
        selection_near = dilate_binary(
            selection_boundary,
            max(1, int(math.ceil(0.40 / resolution))))

    region_lookup = {region.region_id: region for region in regions}
    interface_hints: Dict[int, List[Tuple[np.ndarray, BoundaryType]]] = {
        region.region_id: [] for region in regions}
    expanded_regions = {
        region.region_id: dilate_binary(region.travel_mask, 1)
        for region in regions if region.travel_mask is not None
    }
    interface_expand = max(1, int(math.ceil(0.55 / resolution)))
    for first_index, first in enumerate(regions):
        first_expanded = expanded_regions.get(first.region_id)
        if first_expanded is None:
            continue
        for second in regions[first_index + 1:]:
            if second.travel_mask is None:
                continue
            interface = first_expanded & second.travel_mask
            if not interface.any():
                continue
            for component in _connected_components_fast(interface):
                rows, cols = np.nonzero(component)
                if not cols.size:
                    continue
                span_m = max(
                    (int(rows.max()) - int(rows.min()) + 1) * resolution,
                    (int(cols.max()) - int(cols.min()) + 1) * resolution,
                )
                boundary_type = (
                    BoundaryType.DOOR
                    if span_m <= 1.80 else BoundaryType.PARTITION_SEAM)
                near = dilate_binary(component, interface_expand)
                interface_hints[first.region_id].append(
                    (near, boundary_type))
                interface_hints[second.region_id].append(
                    (near, boundary_type))

    def mask_contains(mask: np.ndarray, point: Point) -> bool:
        row, col = _world_to_cell(
            point, resolution, origin_x, origin_y)
        return bool(0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]
                    and mask[row, col])

    wall_search_min = max(0.18, footprint.half_width_m * 0.60)
    wall_search_max = max(
        0.80,
        footprint.turn_clearance_m
        + max_offset
        + max(BRUSH_LEFT_M, BRUSH_RIGHT_M) * 0.25,
    )
    wall_search_step = max(0.02, resolution * 0.45)

    def wall_distance(point: Point, yaw: float, side: int) -> Optional[float]:
        tangent = (math.cos(yaw), math.sin(yaw))
        normal = (-math.sin(yaw) * side, math.cos(yaw) * side)
        hits: List[float] = []
        for along in (-0.18, 0.0, 0.18):
            sample_origin = (
                point[0] + tangent[0] * along,
                point[1] + tangent[1] * along,
            )
            distance = wall_search_min
            while distance <= wall_search_max + 1e-9:
                probe = (
                    sample_origin[0] + normal[0] * distance,
                    sample_origin[1] + normal[1] * distance,
                )
                row, col = _world_to_cell(
                    probe, resolution, origin_x, origin_y)
                if not (0 <= row < obstacle_mask.shape[0]
                        and 0 <= col < obstacle_mask.shape[1]):
                    break
                if obstacle_mask[row, col]:
                    hits.append(distance)
                    break
                distance += wall_search_step
        return float(np.median(hits)) if len(hits) >= 2 else None

    default_command = CleanerCommand()
    global_profile = [default_command for _ in global_path]
    semantic_failures: List[str] = []

    for segment in segments:
        if not segment.path:
            segment.cleaner_profile = []
            continue
        if segment.kind != "perimeter":
            segment.cleaner_mode = CleanerMode.EDGE_CENTER
            segment.cleaner_offset_m = 0.0
            segment.cleaner_profile = [
                CleanerCommand(cleaning_enabled=segment.kind == "fill")
                for _ in segment.path
            ]
        else:
            _, local_yaws = _explicit_path_yaw_pairs(segment.path)
            raw_commands: List[CleanerCommand] = []
            raw_targets: List[float] = []
            for point, yaw in zip(segment.path, local_yaws):
                left_distance = wall_distance(point, yaw, 1)
                right_distance = wall_distance(point, yaw, -1)
                wall_side = 0
                distance_to_wall: Optional[float] = None
                if left_distance is not None and (
                        right_distance is None
                        or left_distance <= right_distance):
                    wall_side = 1
                    distance_to_wall = left_distance
                elif right_distance is not None:
                    wall_side = -1
                    distance_to_wall = right_distance

                if distance_to_wall is not None:
                    # The physical X stage cannot command positive (left)
                    # offsets.  Preserve the left-wall semantic but never
                    # output an unexecutable actuator command.
                    if wall_side > 0:
                        raw_commands.append(CleanerCommand(
                            mode=CleanerMode.EDGE_IGNORE,
                            offset_m=0.0,
                            boundary_type=BoundaryType.PHYSICAL_WALL,
                            cleaning_enabled=True,
                        ))
                        raw_targets.append(0.0)
                        continue
                    current_edge = (
                        BRUSH_LEFT_M if wall_side > 0 else BRUSH_RIGHT_M)
                    required = max(
                        0.0,
                        distance_to_wall - current_edge
                        - max(float(wall_gap_m), resolution * 0.75),
                    )
                    offset = wall_side * min(max_offset, required)
                    if abs(offset) <= max(0.01, resolution * 0.25):
                        mode = CleanerMode.EDGE_CENTER
                        offset = 0.0
                    else:
                        mode = (
                            CleanerMode.EDGE_LEFT if offset > 0.0
                            else CleanerMode.EDGE_RIGHT)
                    raw_commands.append(CleanerCommand(
                        mode=mode,
                        offset_m=offset,
                        boundary_type=BoundaryType.PHYSICAL_WALL,
                        cleaning_enabled=True,
                    ))
                    raw_targets.append(offset)
                    continue

                boundary_type = BoundaryType.OPEN_BOUNDARY
                for near_mask, hint_type in interface_hints.get(
                        segment.region_id, []):
                    if mask_contains(near_mask, point):
                        boundary_type = hint_type
                        break
                if (boundary_type == BoundaryType.OPEN_BOUNDARY
                        and selection_near is not None
                        and mask_contains(selection_near, point)):
                    boundary_type = BoundaryType.SELECTION_BOUNDARY
                raw_commands.append(CleanerCommand(
                    mode=CleanerMode.EDGE_IGNORE,
                    offset_m=0.0,
                    boundary_type=boundary_type,
                    # V1 preserves the already-certified base geometry.  The
                    # brush therefore still sweeps incidentally at centre while
                    # wall-follow/extension is ignored; a later geometry pass
                    # may remove this non-wall red subsection altogether.
                    cleaning_enabled=True,
                ))
                raw_targets.append(0.0)

            # Purple entry, loop closure, material corners and side changes are
            # zero-offset anchors.  A pair of distance passes then produces a
            # deterministic ramp instead of an instantaneous 0 -> 250 mm jump.
            retract_indices = {0, len(segment.path) - 1}
            for index in range(1, len(segment.path) - 1):
                incoming = _initial_path_yaw(
                    segment.path[index - 1:index + 1])
                outgoing = _initial_path_yaw(
                    segment.path[index:index + 2])
                if (incoming is not None and outgoing is not None
                        and abs(_wrap_angle(outgoing - incoming))
                        >= math.radians(28.0)):
                    retract_indices.add(index)
                previous_target = raw_targets[index - 1]
                current_target = raw_targets[index]
                if previous_target * current_target < -1e-6:
                    retract_indices.update((index - 1, index))
                if (raw_commands[index - 1].boundary_type
                        != raw_commands[index].boundary_type):
                    retract_indices.update((index - 1, index))
            targets = list(raw_targets)
            for index in retract_indices:
                targets[index] = 0.0
            rate = max_offset / transition_distance if max_offset > 0.0 else 0.0
            if rate > 0.0:
                for index in range(1, len(targets)):
                    step_limit = rate * _dist(
                        segment.path[index - 1], segment.path[index])
                    if targets[index - 1] * targets[index] < 0.0:
                        targets[index] = 0.0
                    elif abs(targets[index]) > abs(targets[index - 1]) + step_limit:
                        targets[index] = math.copysign(
                            abs(targets[index - 1]) + step_limit,
                            targets[index])
                for index in range(len(targets) - 2, -1, -1):
                    step_limit = rate * _dist(
                        segment.path[index], segment.path[index + 1])
                    if targets[index] * targets[index + 1] < 0.0:
                        targets[index] = 0.0
                    elif abs(targets[index]) > abs(targets[index + 1]) + step_limit:
                        targets[index] = math.copysign(
                            abs(targets[index + 1]) + step_limit,
                            targets[index])

            commands: List[CleanerCommand] = []
            for index, (point, yaw, base_command, target) in enumerate(zip(
                    segment.path, local_yaws, raw_commands, targets)):
                if base_command.mode == CleanerMode.EDGE_IGNORE:
                    commands.append(base_command)
                    continue
                safe_offset = float(target)
                decrement = max(0.01, resolution * 0.25)
                while (abs(safe_offset) > 1e-6
                        and not _brush_pose_is_free(
                            raw_free_mask, point, yaw, safe_offset,
                            resolution, origin_x, origin_y)):
                    safe_offset = math.copysign(
                        max(0.0, abs(safe_offset) - decrement), safe_offset)
                if not _brush_pose_is_free(
                        raw_free_mask, point, yaw, safe_offset,
                        resolution, origin_x, origin_y):
                    semantic_failures.append(
                        f"region_{segment.region_id}_brush_pose_{index}_unsafe")
                    commands.append(CleanerCommand(
                        mode=CleanerMode.EDGE_IGNORE,
                        boundary_type=base_command.boundary_type,
                        cleaning_enabled=False,
                    ))
                    continue
                if abs(safe_offset) <= 1e-6:
                    mode = (
                        CleanerMode.EDGE_RETRACT
                        if abs(raw_targets[index]) > 1e-6
                        or index in retract_indices
                        else CleanerMode.EDGE_CENTER)
                else:
                    mode = (
                        CleanerMode.EDGE_LEFT if safe_offset > 0.0
                        else CleanerMode.EDGE_RIGHT)
                commands.append(CleanerCommand(
                    mode=mode,
                    offset_m=safe_offset,
                    boundary_type=base_command.boundary_type,
                    cleaning_enabled=True,
                ))
            segment.cleaner_profile = commands
            uniform_modes = {command.mode for command in commands}
            uniform_offsets = {
                round(command.offset_m, 6) for command in commands}
            if len(uniform_modes) == 1:
                segment.cleaner_mode = next(iter(uniform_modes))
            else:
                segment.cleaner_mode = CleanerMode.EDGE_RETRACT
            segment.cleaner_offset_m = (
                next(iter(uniform_offsets)) if len(uniform_offsets) == 1 else 0.0)

        # Transfer geometry can differ after refinement, so use its certified
        # global index span rather than matching points back to the route.
        if segment.kind == "transfer":
            start = max(0, segment.path_start_idx)
            end = min(len(global_profile) - 1, segment.path_end_idx)
            for global_index in range(start, end + 1):
                global_profile[global_index] = CleanerCommand(
                    cleaning_enabled=False)
            continue
        if segment.kind != "perimeter":
            continue

        # Project commands onto the *executed* points.  ``_extend_dedup`` may
        # remove a long chain of sub-15 mm perimeter samples; requiring every
        # discarded local sample to find a global twin incorrectly failed on
        # high-resolution maps.  The global slice is an ordered subsequence, so
        # one monotonic nearest-neighbour walk is exact and linear-time.
        local_index = 0
        global_start = max(0, segment.path_start_idx)
        global_end = min(len(global_path) - 1, segment.path_end_idx)
        mapping_tolerance = max(0.03, resolution * 0.75)
        for global_index in range(global_start, global_end + 1):
            global_point = global_path[global_index]
            while (local_index + 1 < len(segment.path)
                    and _dist(segment.path[local_index + 1], global_point)
                    <= _dist(segment.path[local_index], global_point)):
                local_index += 1
            if _dist(segment.path[local_index], global_point) > mapping_tolerance:
                semantic_failures.append(
                    f"segment_{segment.kind}_{segment.region_id}_profile_unmapped")
                continue
            global_profile[global_index] = segment.cleaner_profile[local_index]

    cleaner_center_path: List[Point] = []
    _, global_departure_yaws = _explicit_path_yaw_pairs(global_path)
    for point, yaw, command in zip(
            global_path, global_departure_yaws, global_profile):
        cleaner_center_path.append((
            point[0] - math.sin(yaw) * command.offset_m,
            point[1] + math.cos(yaw) * command.offset_m,
        ))

    mode_counts: Dict[str, int] = {}
    boundary_counts: Dict[str, int] = {}
    for command in global_profile:
        mode_counts[command.mode.value] = (
            mode_counts.get(command.mode.value, 0) + 1)
        boundary_counts[command.boundary_type.value] = (
            boundary_counts.get(command.boundary_type.value, 0) + 1)
    valid = not semantic_failures and len(global_profile) == len(global_path)
    return (
        global_profile,
        cleaner_center_path,
        valid,
        ";".join(semantic_failures[:12]),
        mode_counts,
        boundary_counts,
    )


def _connected_components_fast(
    mask: np.ndarray,
    min_cells: int = 1,
) -> List[np.ndarray]:
    """Return 4-connected components without iterative full-image dilation.

    ``free_space.largest_component`` deliberately favours compact numpy code,
    but repeatedly invoking its whole-image morphology inside a recursive room
    decomposition is expensive on a 415x725 saved map.  This queue flood visits
    every true cell once and is used only by the offline partitioned planner.
    """
    if not mask.any():
        return []
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        r0, c0 = int(sr), int(sc)
        if seen[r0, c0]:
            continue
        queue = deque([(r0, c0)])
        seen[r0, c0] = True
        cells: List[Cell] = []
        while queue:
            row, col = queue.popleft()
            cells.append((row, col))
            for nr, nc in (
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ):
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if seen[nr, nc] or not mask[nr, nc]:
                    continue
                seen[nr, nc] = True
                queue.append((nr, nc))
        if len(cells) < max(1, int(min_cells)):
            continue
        component = np.zeros_like(mask, dtype=bool)
        rr, cc = zip(*cells)
        component[rr, cc] = True
        components.append(component)
    return components


def _boustrophedon_cells(
    mask: np.ndarray,
    axis: str,
    min_cell_area_m2: float,
    resolution: float,
) -> List[np.ndarray]:
    """Split a connected mask at sweep-line interval split/merge events.

    For horizontal cleaning lanes the sweep advances row by row; for vertical
    lanes the same algorithm runs on the transpose.  A run keeps its cell label
    only across a one-to-one overlap.  When an obstacle causes one run to split
    (or two runs to merge), every outgoing run starts a new cell.  Consequently
    one cell is completely covered before the route crosses the critical line,
    instead of interleaving fragments on opposite sides of an obstacle.
    """
    if not mask.any():
        return []
    work = mask if axis == "x" else mask.T
    labels = np.full(work.shape, -1, dtype=np.int32)
    next_label = 0
    previous: List[Tuple[int, int, int]] = []
    for row in range(work.shape[0]):
        current_runs = list(_true_runs(work[row], 1))
        previous_to_current: List[List[int]] = [
            [] for _ in previous]
        current_to_previous: List[List[int]] = [
            [] for _ in current_runs]
        for current_index, (current_lo, current_hi) in enumerate(current_runs):
            for previous_index, (previous_lo, previous_hi, _) in enumerate(previous):
                if min(current_hi, previous_hi) < max(current_lo, previous_lo):
                    continue
                current_to_previous[current_index].append(previous_index)
                previous_to_current[previous_index].append(current_index)
        next_previous: List[Tuple[int, int, int]] = []
        for current_index, (lo, hi) in enumerate(current_runs):
            linked_previous = current_to_previous[current_index]
            label = -1
            if len(linked_previous) == 1:
                previous_index = linked_previous[0]
                if len(previous_to_current[previous_index]) == 1:
                    label = previous[previous_index][2]
            if label < 0:
                label = next_label
                next_label += 1
            labels[row, lo:hi + 1] = label
            next_previous.append((lo, hi, label))
        previous = next_previous

    cells: List[np.ndarray] = []
    for label in range(next_label):
        cell = labels == label
        if axis != "x":
            cell = cell.T
        if cell.any():
            cells.append(cell)
    if len(cells) <= 1:
        return cells

    # Single-row chips around rasterised furniture should not become their own
    # cleaning phase.  Merge only genuinely small cells into the neighbour with
    # the longest shared boundary; normal BCD cells remain untouched.
    min_cells = max(
        1, int(math.ceil(float(min_cell_area_m2) / resolution ** 2)))
    while len(cells) > 1:
        areas = [int(cell.sum()) for cell in cells]
        small = [index for index, area in enumerate(areas) if area < min_cells]
        if not small:
            break
        source_index = min(small, key=lambda index: areas[index])
        source = cells[source_index]
        expanded = dilate_binary(source, 1)
        target_index = max(
            (index for index in range(len(cells)) if index != source_index),
            key=lambda index: (
                int((expanded & cells[index]).sum()),
                -abs(areas[index] - areas[source_index]),
            ),
        )
        cells[target_index] |= source
        cells.pop(source_index)
    cells.sort(key=lambda cell: int(cell.sum()), reverse=True)
    return cells


def _mask_geometry(mask: np.ndarray
                   ) -> Optional[Tuple[int, Tuple[int, int, int, int], float, int]]:
    """Return ``area_cells, (r0,r1,c0,c1), rectangularity, bbox_cells``."""
    rows, cols = np.nonzero(mask)
    if cols.size == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    area = int(cols.size)
    bbox_area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
    return area, (r0, r1, c0, c1), area / bbox_area, bbox_area


def _principal_axis_stats(mask: np.ndarray) -> Tuple[float, float]:
    """Return the PCA long-axis angle and standard-deviation aspect ratio."""
    rows, cols = np.nonzero(mask)
    if cols.size < 20:
        return 0.0, 1.0
    x = cols.astype(np.float64) - float(cols.mean())
    y = rows.astype(np.float64) - float(rows.mean())
    cxx = float((x * x).mean())
    cyy = float((y * y).mean())
    cxy = float((x * y).mean())
    theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    while theta >= math.pi / 2.0:
        theta -= math.pi
    while theta < -math.pi / 2.0:
        theta += math.pi
    trace = cxx + cyy
    spread = math.hypot(cxx - cyy, 2.0 * cxy)
    major = max(1e-9, 0.5 * (trace + spread))
    minor = max(1e-9, 0.5 * (trace - spread))
    return theta, math.sqrt(major / minor)


def _generate_component_swaths(
    component: np.ndarray,
    bbox_cells: Tuple[int, int, int, int],
    resolution: float,
    origin_x: float,
    origin_y: float,
    swath_spacing_m: float,
    min_swath_m: float,
) -> Tuple[str, float, List[Swath]]:
    """Generate long-edge swaths for one near-rectangular cleaning island.

    Axis-aligned generation remains the default.  PCA rotation is enabled only
    for a clearly elongated component whose long edge differs from both map
    axes by at least 10 degrees.  This handles genuinely rotated rooms without
    reviving the old diagonal-lane failure on irregular L-shaped spaces.
    """
    r0, r1, c0, c1 = bbox_cells
    theta, aspect = _principal_axis_stats(component)
    deviation = min(abs(theta), abs(math.pi / 2.0 - abs(theta)))
    if aspect >= 1.55 and deviation >= math.radians(10.0):
        rotated = _make_rot_frame(
            component, resolution, origin_x, origin_y, theta)
        if rotated is not None:
            rmask, rox, roy, rcx, rcy, cosine, sine = rotated
            rotated_geometry = _mask_geometry(rmask)
            # A long but L-shaped/branched component also has a stable PCA
            # angle; rotating lanes through it creates the old diagonal
            # cross-room pattern.  Require the component to be reasonably
            # rectangle-like in its own principal frame before accepting PCA.
            # Genuine rotated rooms remain near 0.9; axis-aligned rooms whose
            # furniture/no-data bites merely bias PCA fall below 0.60 and keep
            # their wall-aligned long straight lanes.
            if rotated_geometry is None or rotated_geometry[2] < 0.60:
                rotated = None
        if rotated is not None:
            rmask, rox, roy, rcx, rcy, cosine, sine = rotated
            raw_rotated = _generate_swaths_from_mask(
                rmask, resolution, rox, roy,
                swath_spacing_m=swath_spacing_m,
                axis="x", min_swath_m=min_swath_m,
            )

            def to_world(point: Point) -> Point:
                return (
                    rcx + cosine * point[0] - sine * point[1],
                    rcy + sine * point[0] + cosine * point[1],
                )

            clipped: List[Swath] = []
            for start, end in raw_rotated:
                segment = _clip_swath_to_mask(
                    component, to_world(start), to_world(end),
                    resolution, origin_x, origin_y, min_swath_m)
                if segment is not None:
                    clipped.append(segment)
            if clipped:
                return "rotated", theta, clipped

    axis = "x" if (c1 - c0) >= (r1 - r0) else "y"
    swaths = _generate_swaths_from_mask(
        component[r0:r1 + 1, c0:c1 + 1], resolution,
        origin_x + c0 * resolution,
        origin_y + r0 * resolution,
        swath_spacing_m=swath_spacing_m,
        axis=axis, min_swath_m=min_swath_m,
    )
    return axis, (0.0 if axis == "x" else math.pi / 2.0), swaths


def _order_component_swaths(
    swaths: Sequence[Swath],
    start: Point,
    axis: str,
    angle: float,
    spacing_m: float,
    ascending: bool = True,
    initial_forward: Optional[bool] = None,
) -> List[Swath]:
    """Apply the same one-way lane progression in axis or rotated frames."""
    if axis != "rotated":
        return _order_swaths_monotonic(
            swaths, start, axis, spacing_m, ascending=ascending,
            initial_forward=initial_forward)
    cosine, sine = math.cos(angle), math.sin(angle)

    def to_rot(point: Point) -> Point:
        return (
            cosine * point[0] + sine * point[1],
            -sine * point[0] + cosine * point[1],
        )

    def to_world(point: Point) -> Point:
        return (
            cosine * point[0] - sine * point[1],
            sine * point[0] + cosine * point[1],
        )

    rotated_swaths = [(to_rot(a), to_rot(b)) for a, b in swaths]
    ordered = _order_swaths_monotonic(
        rotated_swaths, to_rot(start), "x", spacing_m,
        ascending=ascending, initial_forward=initial_forward)
    return [(to_world(a), to_world(b)) for a, b in ordered]


def _best_neck_cut(
    crop: np.ndarray,
    resolution: float,
    transpose: bool,
    min_child_area_m2: float,
    min_child_span_m: float,
    max_portal_width_m: float,
) -> Optional[Tuple[float, int, float, int]]:
    """Score guillotine cuts at narrow passages in one local mask.

    Rows are the candidate split direction.  Transposition therefore evaluates
    a vertical world cut with the same code.  Only a genuinely narrow bridge,
    or a bridge occupying a small fraction of the room width, is eligible;
    this prevents the old failure where a long corridor was split into several
    parallel strips merely because that improved bounding-box rectangularity.
    """
    work = crop.T if transpose else crop
    n_rows, n_cols = work.shape
    if n_rows < 3 or n_cols < 2:
        return None
    row_counts = work.sum(axis=1).astype(np.int64)
    if not row_counts.any():
        return None
    row_min = np.full(n_rows, n_cols, dtype=np.int64)
    row_max = np.full(n_rows, -1, dtype=np.int64)
    for idx in range(n_rows):
        occupied = np.flatnonzero(work[idx])
        if occupied.size:
            row_min[idx] = int(occupied[0])
            row_max[idx] = int(occupied[-1])
    area_prefix = np.cumsum(row_counts)
    area_total = int(area_prefix[-1])
    prefix_min = np.minimum.accumulate(row_min)
    prefix_max = np.maximum.accumulate(row_max)
    suffix_min = np.minimum.accumulate(row_min[::-1])[::-1]
    suffix_max = np.maximum.accumulate(row_max[::-1])[::-1]
    min_area_cells = max(1, int(math.ceil(min_child_area_m2 / resolution ** 2)))
    min_span_cells = max(2, int(math.ceil(min_child_span_m / resolution)))
    parent_bbox_area = n_rows * n_cols
    best: Optional[Tuple[float, int, float, int]] = None
    for cut in range(min_span_cells - 1, n_rows - min_span_cells):
        left_area = int(area_prefix[cut])
        right_area = area_total - left_area
        if left_area < min_area_cells or right_area < min_area_cells:
            continue
        left_width = int(prefix_max[cut] - prefix_min[cut] + 1)
        right_width = int(suffix_max[cut + 1] - suffix_min[cut + 1] + 1)
        if left_width <= 0 or right_width <= 0:
            continue
        left_bbox = (cut + 1) * left_width
        right_bbox = (n_rows - cut - 1) * right_width
        improvement = (
            parent_bbox_area - left_bbox - right_bbox
        ) / max(1, parent_bbox_area)
        bridge_cells = int(np.sum(work[cut] & work[cut + 1]))
        bridge_ratio = bridge_cells / max(1, n_cols)
        if (bridge_cells * resolution > max_portal_width_m
                and bridge_ratio > 0.42):
            continue
        balance = min(left_area, right_area) / max(1, area_total)
        score = improvement + 0.30 * (1.0 - bridge_ratio) + 0.08 * balance
        candidate = (score, cut, improvement, bridge_cells)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def _partition_travel_component(
    travel_component: np.ndarray,
    resolution: float,
    min_region_area_m2: float = 5.0,
    max_regions: int = 12,
) -> List[np.ndarray]:
    """Split reachable travel space at doors/necks into room-like masks.

    The decomposition operates on the 0.34m travel topology, before the larger
    turn-safe erosion.  Consequently doors remain visible and every resulting
    room can later receive its own ``fill -> perimeter`` phase.  Long shapes are
    cut only perpendicular to their dominant dimension, so a central corridor
    remains a cleanable longitudinal region rather than parallel slivers.
    """
    min_child_area = max(1.0, float(min_region_area_m2))

    def recurse(mask: np.ndarray, depth: int = 0) -> List[np.ndarray]:
        geometry = _mask_geometry(mask)
        if geometry is None:
            return []
        area, (r0, r1, c0, c1), rectangularity, _ = geometry
        height_m = (r1 - r0 + 1) * resolution
        width_m = (c1 - c0 + 1) * resolution
        if (
            depth >= 8
            or area * resolution ** 2 < min_child_area * 2.7
            or min(height_m, width_m) < 2.0
        ):
            return [mask]
        crop = mask[r0:r1 + 1, c0:c1 + 1]
        if height_m > width_m * 1.20:
            orientations = (False,)       # horizontal cut across a tall region
        elif width_m > height_m * 1.20:
            orientations = (True,)        # vertical cut across a wide region
        else:
            orientations = (False, True)
        candidates: List[Tuple[Tuple[float, int, float, int], bool]] = []
        for transpose in orientations:
            candidate = _best_neck_cut(
                crop, resolution, transpose,
                min_child_area_m2=min_child_area,
                min_child_span_m=1.35,
                max_portal_width_m=3.0,
            )
            if candidate is not None:
                candidates.append((candidate, transpose))
        # A tall corridor with several rooms along one side has multiple
        # separated door bridges on a *vertical* wall (and vice versa).  The
        # dominant-axis rule above intentionally protects ordinary long rooms,
        # but would merge that whole row of rooms into the corridor.  Admit the
        # perpendicular cut only when it exposes at least two distinct portal
        # runs and at least three connected children; a single doorway/noisy
        # notch cannot trigger this exception.
        if len(orientations) == 1:
            perpendicular = not orientations[0]
            candidate = _best_neck_cut(
                crop, resolution, perpendicular,
                min_child_area_m2=min_child_area,
                min_child_span_m=1.35,
                max_portal_width_m=3.0,
            )
            if candidate is not None:
                _, candidate_cut, _, _ = candidate
                work = crop.T if perpendicular else crop
                bridge = work[candidate_cut] & work[candidate_cut + 1]
                portal_runs = sum(
                    1 for index, value in enumerate(bridge)
                    if value and (index == 0 or not bridge[index - 1]))
                first_work = work.copy()
                first_work[candidate_cut + 1:, :] = False
                second_work = work.copy()
                second_work[:candidate_cut + 1, :] = False
                substantial_cells = max(
                    1, int(math.ceil(min_child_area / resolution ** 2)))
                substantial_children = _connected_components_fast(
                    first_work, min_cells=substantial_cells)
                substantial_children.extend(_connected_components_fast(
                    second_work, min_cells=substantial_cells))
                children_are_room_like = all(
                    geometry is not None and geometry[2] >= 0.78
                    for geometry in (
                        _mask_geometry(child)
                        for child in substantial_children))
                if (portal_runs >= 2
                        and len(substantial_children) >= 3
                        and children_are_room_like):
                    candidates.append((candidate, perpendicular))
        chosen_cut: Optional[
            Tuple[Tuple[float, int, float, int], bool]
        ] = max(candidates, key=lambda item: item[0][0]) if candidates else None
        if chosen_cut is not None:
            _, _, neck_improvement, neck_bridge = chosen_cut[0]
            if neck_improvement < 0.025 and neck_bridge * resolution > 1.50:
                chosen_cut = None

        # A broad L/T-shaped opening is not a door, so the neck detector must
        # reject it.  It still needs two near-rectangular cleaning cells.  Only
        # for a clearly non-rectangular parent, try one high-value guillotine
        # cut in both directions.  The 8% bounding-box waste reduction and
        # child-size gates prevent the old BCD failure that shredded a normal
        # corridor into many slivers.
        if chosen_cut is None and rectangularity < 0.82:
            rectangle_candidates: List[
                Tuple[Tuple[float, int, float, int], bool]
            ] = []
            for transpose_candidate in (False, True):
                candidate = _best_neck_cut(
                    crop, resolution, transpose_candidate,
                    min_child_area_m2=min_child_area,
                    min_child_span_m=1.35,
                    max_portal_width_m=float("inf"),
                )
                if candidate is not None:
                    rectangle_candidates.append(
                        (candidate, transpose_candidate))
            if rectangle_candidates:
                rectangle_cut = max(
                    rectangle_candidates,
                    key=lambda item: (item[0][2], item[0][0]),
                )
                if rectangle_cut[0][2] >= 0.08:
                    chosen_cut = rectangle_cut
        if chosen_cut is None:
            return [mask]
        (_, local_cut, _, _), transpose = chosen_cut
        first = np.zeros_like(mask, dtype=bool)
        second = np.zeros_like(mask, dtype=bool)
        if transpose:
            cut = c0 + local_cut
            first[:, :cut + 1] = mask[:, :cut + 1]
            second[:, cut + 1:] = mask[:, cut + 1:]
        else:
            cut = r0 + local_cut
            first[:cut + 1, :] = mask[:cut + 1, :]
            second[cut + 1:, :] = mask[cut + 1:, :]
        children = _connected_components_fast(first)
        children.extend(_connected_components_fast(second))
        if len(children) < 2:
            return [mask]
        result: List[np.ndarray] = []
        for child in children:
            result.extend(recurse(child, depth + 1))
        return result

    regions = recurse(travel_component)
    if not regions:
        return []

    # Small chips arise around furniture/wall raster noise after a cut.  Merge
    # each chip into the region sharing the longest 4-neighbour boundary.  The
    # union stays connected and no reachable travel cells are silently dropped.
    min_cells = max(1, int(math.ceil(min_region_area_m2 / resolution ** 2)))
    while len(regions) > 1:
        areas = [int(region.sum()) for region in regions]
        small_candidates = [i for i, area in enumerate(areas) if area < min_cells]
        if not small_candidates and len(regions) <= max_regions:
            break
        source_i = min(
            small_candidates if small_candidates else range(len(regions)),
            key=lambda idx: areas[idx],
        )
        source = regions[source_i]
        expanded = dilate_binary(source, 1)
        source_geometry = _mask_geometry(source)
        source_center = (0.0, 0.0)
        if source_geometry is not None:
            _, (sr0, sr1, sc0, sc1), _, _ = source_geometry
            source_center = ((sr0 + sr1) * 0.5, (sc0 + sc1) * 0.5)
        best_j = -1
        best_key = (-1, float("-inf"))
        for target_i, target in enumerate(regions):
            if target_i == source_i:
                continue
            shared = int(np.sum(expanded & target))
            target_geometry = _mask_geometry(target)
            if target_geometry is None:
                continue
            _, (tr0, tr1, tc0, tc1), _, _ = target_geometry
            distance = math.hypot(
                source_center[0] - (tr0 + tr1) * 0.5,
                source_center[1] - (tc0 + tc1) * 0.5,
            )
            key = (shared, -distance)
            if key > best_key:
                best_key = key
                best_j = target_i
        if best_j < 0:
            break
        regions[best_j] = regions[best_j] | source
        regions.pop(source_i)
    regions.sort(key=lambda region: int(region.sum()), reverse=True)
    return regions


def _order_swaths_monotonic(
    swaths: Sequence[Swath],
    start: Point,
    axis: str,
    spacing_m: float,
    ascending: bool = True,
    initial_forward: Optional[bool] = None,
) -> List[Swath]:
    """Order lanes once across their short axis, with no skip-and-return pass."""
    if not swaths:
        return []
    perp = 1 if axis == "x" else 0
    along = 0 if axis == "x" else 1
    tolerance = max(1e-6, float(spacing_m) * 0.50)

    def lane_coordinate(swath: Swath) -> float:
        return 0.5 * (swath[0][perp] + swath[1][perp])

    items = sorted(swaths, key=lane_coordinate, reverse=not ascending)
    lanes: List[List[Swath]] = [[items[0]]]
    for swath in items[1:]:
        if abs(lane_coordinate(swath) - lane_coordinate(lanes[-1][-1])) <= tolerance:
            lanes[-1].append(swath)
        else:
            lanes.append([swath])
    ordered: List[Swath] = []
    forward: Optional[bool] = initial_forward
    for lane in lanes:
        lane_sorted = sorted(
            lane, key=lambda segment: min(segment[0][along], segment[1][along]))
        if forward is None:
            low = min(min(a[along], b[along]) for a, b in lane_sorted)
            high = max(max(a[along], b[along]) for a, b in lane_sorted)
            forward = abs(start[along] - low) <= abs(start[along] - high)
        sequence = lane_sorted if forward else list(reversed(lane_sorted))
        for a, b in sequence:
            if forward and a[along] > b[along]:
                a, b = b, a
            elif not forward and a[along] < b[along]:
                a, b = b, a
            ordered.append((a, b))
        forward = not forward
    return ordered


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def _explicit_path_yaw_pairs(
    path: Sequence[Point],
    fallback_yaw: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """Return explicit arrival/departure yaw for every path point.

    A single orientation is insufficient at a stop-and-turn vertex.  Arrival
    follows the nearest non-duplicate predecessor; departure follows the
    nearest non-duplicate successor.  Endpoints inherit the only available
    tangent.  Duplicate raster points are tolerated defensively.
    """
    points = list(path)
    if not points:
        return [], []
    arrivals: List[float] = []
    departures: List[float] = []
    for index, point in enumerate(points):
        previous = next((
            points[candidate]
            for candidate in range(index - 1, -1, -1)
            if _dist(points[candidate], point) > 1e-8
        ), None)
        following = next((
            points[candidate]
            for candidate in range(index + 1, len(points))
            if _dist(points[candidate], point) > 1e-8
        ), None)
        if previous is not None:
            arrival = math.atan2(
                point[1] - previous[1], point[0] - previous[0])
        elif following is not None:
            arrival = math.atan2(
                following[1] - point[1], following[0] - point[0])
        else:
            arrival = float(fallback_yaw)
        if following is not None:
            departure = math.atan2(
                following[1] - point[1], following[0] - point[0])
        else:
            departure = arrival
        arrivals.append(_wrap_angle(arrival))
        departures.append(_wrap_angle(departure))
    return arrivals, departures


def _initial_path_yaw(
    path: Sequence[Point],
    fallback: Optional[float] = None,
) -> Optional[float]:
    for start, end in zip(path, path[1:]):
        if _dist(start, end) > 1e-7:
            return math.atan2(end[1] - start[1], end[0] - start[0])
    return fallback


def _terminal_path_yaw(
    path: Sequence[Point],
    fallback: Optional[float] = None,
) -> Optional[float]:
    for start, end in reversed(list(zip(path, path[1:]))):
        if _dist(start, end) > 1e-7:
            return math.atan2(end[1] - start[1], end[0] - start[0])
    return fallback


def _path_yaws_lookahead(
    path: Sequence[Point],
    lookahead_m: float = 0.55,
) -> List[float]:
    """Infer the forward body yaw used for offline swept-footprint checks."""
    if not path:
        return []
    if len(path) == 1:
        return [0.0]
    yaws: List[float] = []
    for index, point in enumerate(path):
        walked = 0.0
        target: Optional[Point] = None
        cursor = point
        for nxt in path[index + 1:]:
            step = _dist(cursor, nxt)
            walked += step
            cursor = nxt
            if walked >= lookahead_m and _dist(point, nxt) > 1e-6:
                target = nxt
                break
        if target is not None:
            yaws.append(math.atan2(target[1] - point[1], target[0] - point[0]))
            continue
        # Near the path tail, keep facing forward by looking from a prior point
        # toward the current point; do not flip the robot 180 degrees.
        walked = 0.0
        source: Optional[Point] = None
        cursor = point
        for previous in reversed(path[:index]):
            step = _dist(cursor, previous)
            walked += step
            cursor = previous
            if walked >= lookahead_m and _dist(previous, point) > 1e-6:
                source = previous
                break
        if source is None:
            source = path[max(0, index - 1)]
        if _dist(source, point) > 1e-6:
            yaws.append(math.atan2(point[1] - source[1], point[0] - source[0]))
        elif yaws:
            yaws.append(yaws[-1])
        else:
            yaws.append(0.0)
    return yaws


def _footprint_samples(
    footprint: CoverageFootprint,
    resolution: float,
) -> np.ndarray:
    """Dense body-frame samples of the margin-expanded TD25A rectangle."""
    step = max(0.015, resolution * 0.40)
    rear = footprint.rear_m + footprint.tracking_margin_m
    front = footprint.front_m + footprint.tracking_margin_m
    half_width = footprint.half_width_m + footprint.tracking_margin_m
    xs = np.arange(-rear, front + step * 0.5, step, dtype=np.float64)
    ys = np.arange(-half_width, half_width + step * 0.5, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _footprint_pose_is_free(
    raw_free_mask: np.ndarray,
    position: Point,
    yaw: float,
    footprint_samples: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> bool:
    """Conservatively sample the complete, asymmetric body at one pose."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    wx = position[0] + cosine * footprint_samples[:, 0] - sine * footprint_samples[:, 1]
    wy = position[1] + sine * footprint_samples[:, 0] + cosine * footprint_samples[:, 1]
    cols = np.floor((wx - origin_x) / resolution).astype(np.int64)
    rows = np.floor((wy - origin_y) / resolution).astype(np.int64)
    valid = (
        (rows >= 0) & (rows < raw_free_mask.shape[0])
        & (cols >= 0) & (cols < raw_free_mask.shape[1])
    )
    if not bool(valid.all()):
        return False
    return bool(raw_free_mask[rows, cols].all())


def _shortest_rotation_is_free(
    raw_free_mask: np.ndarray,
    position: Point,
    start_yaw: float,
    goal_yaw: float,
    footprint_samples: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> bool:
    """Check the rotation direction actually chosen from two pose headings.

    A quaternion contains no instruction to rotate the long way around.  Nav2's
    RotationShim therefore uses the wrapped shortest angular distance.  Safety
    validation must certify that same direction; accepting a safe 270 degree
    alternative when the commanded 90 degree turn clips a wall is unsound.
    """
    delta = _wrap_angle(float(goal_yaw) - float(start_yaw))
    count = max(1, int(math.ceil(abs(delta) / math.radians(5.0))))
    return all(_footprint_pose_is_free(
        raw_free_mask,
        position,
        float(start_yaw) + delta * index / count,
        footprint_samples,
        resolution,
        origin_x,
        origin_y,
    ) for index in range(count + 1))


def validate_explicit_pose_transition(
    raw_free_mask: np.ndarray,
    start: Point,
    start_yaw: float,
    goal: Point,
    goal_yaw: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
    footprint: Optional[CoverageFootprint] = None,
) -> Tuple[bool, int]:
    """Validate ``rotate shortest -> translate -> rotate shortest``.

    This is the execution contract for the short current-pose lead-in added in
    front of a precomputed partitioned FollowPath slice.  The translation keeps
    the line tangent yaw; both endpoint rotations use the exact asymmetric body
    and the same <=5 degree sampling as the offline planner.
    """
    body = footprint or CoverageFootprint()
    samples = _footprint_samples(body, resolution)
    dx = float(goal[0]) - float(start[0])
    dy = float(goal[1]) - float(start[1])
    distance = math.hypot(dx, dy)
    if distance <= 1e-7:
        valid = _shortest_rotation_is_free(
            raw_free_mask, start, start_yaw, goal_yaw, samples,
            resolution, origin_x, origin_y)
        return valid, 0 if valid else 1
    travel_yaw = math.atan2(dy, dx)
    violations = 0
    if not _shortest_rotation_is_free(
            raw_free_mask, start, start_yaw, travel_yaw, samples,
            resolution, origin_x, origin_y):
        violations += 1
    translation_step = max(0.02, resolution * 0.40)
    count = max(1, int(math.ceil(distance / translation_step)))
    for index in range(count + 1):
        ratio = index / count
        point = (
            float(start[0]) + dx * ratio,
            float(start[1]) + dy * ratio,
        )
        if not _footprint_pose_is_free(
                raw_free_mask, point, travel_yaw, samples,
                resolution, origin_x, origin_y):
            violations += 1
    if not _shortest_rotation_is_free(
            raw_free_mask, goal, travel_yaw, goal_yaw, samples,
            resolution, origin_x, origin_y):
        violations += 1
    return violations == 0, violations


def validate_footprint_path(
    raw_free_mask: np.ndarray,
    path: Sequence[Point],
    resolution: float,
    origin_x: float,
    origin_y: float,
    footprint: Optional[CoverageFootprint] = None,
    lookahead_m: float = 0.55,
    first_yaw_immediate: bool = False,
) -> Tuple[bool, int]:
    """Validate translation and heading changes using the real TD25A body.

    The path is sampled at at most half a map cell.  Heading uses the same
    0.55m forward-look convention as the execution-side orientation logic, and
    every change in heading is interpolated in <=5 degree increments so the
    long rear overhang cannot sweep through a wall between two valid poses.
    """
    if not path:
        return False, 0
    body = footprint or CoverageFootprint()
    samples = _footprint_samples(body, resolution)
    yaws = _path_yaws_lookahead(path, lookahead_m=lookahead_m)
    if first_yaw_immediate and len(path) >= 2:
        first_target = next((
            point for point in path[1:]
            if _dist(path[0], point) > 1e-7
        ), None)
        if first_target is not None:
            yaws[0] = math.atan2(
                first_target[1] - path[0][1],
                first_target[0] - path[0][0])
    violation_count = 0
    if not _footprint_pose_is_free(
            raw_free_mask, path[0], yaws[0], samples,
            resolution, origin_x, origin_y):
        violation_count += 1
    translation_step = max(0.02, resolution * 0.50)
    yaw_step = math.radians(5.0)
    for index, (start, goal) in enumerate(zip(path, path[1:])):
        distance = _dist(start, goal)
        yaw_delta = _wrap_angle(yaws[index + 1] - yaws[index])
        count = max(
            1,
            int(math.ceil(distance / translation_step)),
            int(math.ceil(abs(yaw_delta) / yaw_step)),
        )
        for sample_index in range(1, count + 1):
            ratio = sample_index / count
            position = (
                start[0] + (goal[0] - start[0]) * ratio,
                start[1] + (goal[1] - start[1]) * ratio,
            )
            yaw = yaws[index] + yaw_delta * ratio
            if not _footprint_pose_is_free(
                    raw_free_mask, position, yaw, samples,
                    resolution, origin_x, origin_y):
                violation_count += 1
    return violation_count == 0, violation_count


def validate_explicit_yaw_polyline(
    raw_free_mask: np.ndarray,
    path: Sequence[Point],
    resolution: float,
    origin_x: float,
    origin_y: float,
    footprint: Optional[CoverageFootprint] = None,
    allow_staged_long_rotation: bool = False,
) -> Tuple[bool, int]:
    """Validate tangent poses and exact-body corner rotations.

    The normal execution contract uses Nav2's wrapped shortest turn.  The
    private connector-search helper is the only caller allowed to enable a
    staged long-rotation feasibility probe; final partition validation never
    enables it.
    """
    if len(path) < 2:
        return False, 0
    body = footprint or CoverageFootprint()
    samples = _footprint_samples(body, resolution)
    # Remove duplicate points but retain every actual change of segment heading.
    points: List[Point] = []
    for point in path:
        if not points or _dist(points[-1], point) > 1e-7:
            points.append(point)
    if len(points) < 2:
        return False, 0
    yaws = [
        math.atan2(end[1] - start[1], end[0] - start[0])
        for start, end in zip(points, points[1:])
    ]
    violations = 0
    translation_step = max(0.02, resolution * 0.40)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        yaw = yaws[index]
        count = max(1, int(math.ceil(_dist(start, end) / translation_step)))
        for sample_index in range(count + 1):
            ratio = sample_index / count
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            if not _footprint_pose_is_free(
                    raw_free_mask, point, yaw, samples,
                    resolution, origin_x, origin_y):
                violations += 1
        if index + 1 >= len(yaws):
            continue
        next_yaw = yaws[index + 1]
        shortest = _wrap_angle(next_yaw - yaw)
        candidate_deltas = [shortest]
        if allow_staged_long_rotation and abs(shortest) > 1e-6:
            candidate_deltas.append(
                shortest - math.copysign(2.0 * math.pi, shortest))
        rotation_feasible = False
        for delta in candidate_deltas:
            count = max(
                1, int(math.ceil(abs(delta) / math.radians(5.0))))
            if all(_footprint_pose_is_free(
                    raw_free_mask, end, yaw + delta * step / count, samples,
                    resolution, origin_x, origin_y)
                   for step in range(count + 1)):
                rotation_feasible = True
                break
        if not rotation_feasible:
            violations += 1
    return violations == 0, violations


def _validate_connector_candidate_polyline(
    raw_free_mask: np.ndarray,
    path: Sequence[Point],
    resolution: float,
    origin_x: float,
    origin_y: float,
    footprint: Optional[CoverageFootprint] = None,
) -> Tuple[bool, int]:
    """Cheap search filter; the final execution validator remains strict.

    A continuous rounded connector may geometrically sweep a corner without
    stopping at its vertex.  During candidate generation we may retain a path
    when either in-place sweep is clear, then let
    ``_validate_partitioned_execution_path`` certify its real continuous
    slices and shortest hard-stop rotations.  This helper must never be used as
    the execution certificate itself.
    """
    return validate_explicit_yaw_polyline(
        raw_free_mask, path, resolution, origin_x, origin_y,
        footprint=footprint, allow_staged_long_rotation=True)


def _validate_partitioned_execution_path(
    raw_free_mask: np.ndarray,
    path: Sequence[Point],
    hard_stop_indices: Sequence[int],
    resolution: float,
    origin_x: float,
    origin_y: float,
    footprint: CoverageFootprint,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Tuple[bool, int]:
    """Validate the same continuous slices and stop rotations sent to Nav2.

    ``RosBridge.orient_path_for_execution`` assigns the immediate tangent yaw
    inside each already-simplified FollowPath slice.  Curves therefore change
    heading while translating; only an explicit hard stop causes RotationShim
    to turn in place.  This routine mirrors that split contract and still sweeps
    the full asymmetric body through every actual stop rotation.
    """
    if len(path) < 2:
        return False, 0
    last_index = len(path) - 1
    stops = sorted({
        int(index) for index in hard_stop_indices
        if 0 < int(index) < last_index
    })
    translation_violations = 0
    rotation_violations = 0
    bad_slices: List[Tuple[object, ...]] = []
    bad_rotation_stops: List[int] = []
    slice_start = 0
    for slice_end in stops + [last_index]:
        if slice_end <= slice_start:
            continue
        _, slice_violations = validate_footprint_path(
            raw_free_mask,
            path[slice_start:slice_end + 1],
            resolution,
            origin_x,
            origin_y,
            footprint=footprint,
            lookahead_m=0.0,
            first_yaw_immediate=True,
        )
        translation_violations += slice_violations
        if slice_violations:
            bad_slices.append((
                slice_start,
                slice_end,
                slice_violations,
                tuple(round(value, 2) for value in path[slice_start]),
                tuple(round(value, 2) for value in path[slice_end]),
            ))
        slice_start = slice_end

    body_samples = _footprint_samples(footprint, resolution)
    for stop in stops:
        previous_index = stop - 1
        while (previous_index >= 0
               and _dist(path[previous_index], path[stop]) <= 1e-7):
            previous_index -= 1
        next_index = stop + 1
        while (next_index <= last_index
               and _dist(path[next_index], path[stop]) <= 1e-7):
            next_index += 1
        if previous_index < 0 or next_index > last_index:
            continue
        incoming_yaw = math.atan2(
            path[stop][1] - path[previous_index][1],
            path[stop][0] - path[previous_index][0])
        outgoing_yaw = math.atan2(
            path[next_index][1] - path[stop][1],
            path[next_index][0] - path[stop][0])
        if not _shortest_rotation_is_free(
                raw_free_mask, path[stop], incoming_yaw, outgoing_yaw,
                body_samples, resolution, origin_x, origin_y):
            rotation_violations += 1
            bad_rotation_stops.append(stop)
    violations = translation_violations + rotation_violations
    if diagnostics is not None:
        diagnostics["translation"] = translation_violations
        diagnostics["rotation"] = rotation_violations
        diagnostics["bad_slices"] = bad_slices
        diagnostics["bad_rotation_stops"] = bad_rotation_stops
    return violations == 0, violations


def _astar_se2_lattice(
    raw_free_mask: np.ndarray,
    allowed_center_mask: np.ndarray,
    turn_safe_mask: np.ndarray,
    rotation_safe_mask: Optional[np.ndarray],
    start: Point,
    goal: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
    footprint: CoverageFootprint,
    path_step_m: float,
    start_yaw: Optional[float] = None,
    goal_yaw: Optional[float] = None,
    guide_path: Optional[Sequence[Point]] = None,
    max_expansions: int = 180_000,
    stencil_table_cache: Optional[Dict[Tuple[object, ...], object]] = None,
    penalty_mask: Optional[np.ndarray] = None,
) -> List[Point]:
    """Body-aware 16-heading A* used only when a 2-D connector is unsafe.

    In addition to the four cardinal and four diagonal headings, shallow
    ``1:2``/``2:1`` primitives represent doors and corridors whose axes are not
    exact multiples of 45 degrees.  Every rotation and translation primitive
    sweeps the full asymmetric footprint.  No reverse action is provided,
    matching the requested "avoid going back" behaviour.
    """
    start_cell = _snap_to_free(
        allowed_center_mask,
        _world_to_cell(start, resolution, origin_x, origin_y),
        max_radius=max(20, int(math.ceil(1.5 / resolution))),
    )
    goal_cell = _snap_to_free(
        allowed_center_mask,
        _world_to_cell(goal, resolution, origin_x, origin_y),
        max_radius=max(20, int(math.ceil(1.5 / resolution))),
    )
    if start_cell is None or goal_cell is None:
        return []
    height, width = allowed_center_mask.shape
    roi_pad = max(8, int(math.ceil(2.0 / resolution)))
    guide_cells = [
        _world_to_cell(point, resolution, origin_x, origin_y)
        for point in (guide_path or (start, goal))
    ]
    rows = [start_cell[0], goal_cell[0]] + [cell[0] for cell in guide_cells]
    cols = [start_cell[1], goal_cell[1]] + [cell[1] for cell in guide_cells]
    roi_r0 = max(0, min(rows) - roi_pad)
    roi_r1 = min(height - 1, max(rows) + roi_pad)
    roi_c0 = max(0, min(cols) - roi_pad)
    roi_c1 = min(width - 1, max(cols) + roi_pad)

    directions = (
        (0, 1), (1, 2), (1, 1), (2, 1),
        (1, 0), (2, -1), (1, -1), (1, -2),
        (0, -1), (-1, -2), (-1, -1), (-2, -1),
        (-1, 0), (-2, 1), (-1, 1), (-1, 2),
    )
    heading_yaws = tuple(math.atan2(dr, dc) for dr, dc in directions)
    heading_count = len(directions)
    pose_cache: Dict[Tuple[int, int, int], bool] = {}
    primitive_cache: Dict[Tuple[int, int, int, int], bool] = {}

    # Flattening every stencil once avoids allocating separate row/column
    # index arrays and recomputing four extrema at every A* expansion.  On the
    # Jetson this collision predicate is the dominant planning hot path.
    Stencil = Tuple[np.ndarray, int, int, int, int]

    table_key: Tuple[object, ...] = (
        width, round(float(resolution), 9),
        round(float(footprint.front_m), 6),
        round(float(footprint.rear_m), 6),
        round(float(footprint.half_width_m), 6),
    )
    tables = (stencil_table_cache.get(table_key)
              if stencil_table_cache is not None else None)
    if tables is None:
        body_samples = _footprint_samples(footprint, resolution)

        def swept_offsets(
            relative_poses: Sequence[Tuple[float, float, float]],
        ) -> Stencil:
            """Raster offsets and bounds touched by cell-centred body poses."""
            chunks: List[np.ndarray] = []
            for tx, ty, yaw in relative_poses:
                cosine = math.cos(yaw)
                sine = math.sin(yaw)
                wx = (tx + cosine * body_samples[:, 0]
                      - sine * body_samples[:, 1])
                wy = (ty + sine * body_samples[:, 0]
                      + cosine * body_samples[:, 1])
                cols = np.floor(wx / resolution + 0.5).astype(np.int32)
                rows = np.floor(wy / resolution + 0.5).astype(np.int32)
                chunks.append(np.column_stack((rows, cols)))
            offsets = np.unique(np.concatenate(chunks, axis=0), axis=0)
            row_offsets = offsets[:, 0]
            col_offsets = offsets[:, 1]
            return (
                row_offsets.astype(np.int64) * width
                + col_offsets.astype(np.int64),
                int(row_offsets.min()), int(row_offsets.max()),
                int(col_offsets.min()), int(col_offsets.max()),
            )

        pose_stencils = [
            swept_offsets(((0.0, 0.0, yaw),)) for yaw in heading_yaws]
        rotation_stencils: Dict[Tuple[int, int], Stencil] = {}
        move_stencils: Dict[int, Stencil] = {}
        move_center_flat_offsets: Dict[int, np.ndarray] = {}
        rotation_costs: Dict[Tuple[int, int], float] = {}
        for heading, (dr, dc) in enumerate(directions):
            for delta in (-1, 1):
                next_heading = (heading + delta) % heading_count
                yaw_delta = _wrap_angle(
                    heading_yaws[next_heading] - heading_yaws[heading])
                sample_count = max(
                    2, int(math.ceil(abs(yaw_delta) / math.radians(5.0))))
                rotation_stencils[(heading, delta)] = swept_offsets(tuple(
                    (0.0, 0.0, heading_yaws[heading]
                     + yaw_delta * index / sample_count)
                    for index in range(sample_count + 1)))
                rotation_costs[(heading, delta)] = (
                    0.45 * abs(yaw_delta) / (math.pi / 4.0))
            move_stencils[heading] = swept_offsets(tuple(
                (dc * resolution * ratio, dr * resolution * ratio,
                 heading_yaws[heading])
                for ratio in np.linspace(0.0, 1.0, 9)))
            centre_offsets = np.unique(np.asarray([
                (int(round(dr * ratio)), int(round(dc * ratio)))
                for ratio in np.linspace(0.0, 1.0, 9)
            ], dtype=np.int32), axis=0)
            move_center_flat_offsets[heading] = (
                centre_offsets[:, 0].astype(np.int64) * width
                + centre_offsets[:, 1].astype(np.int64))
        tables = (
            pose_stencils, rotation_stencils, move_stencils,
            move_center_flat_offsets, rotation_costs)
        if stencil_table_cache is not None:
            stencil_table_cache[table_key] = tables
    else:
        (pose_stencils, rotation_stencils, move_stencils,
         move_center_flat_offsets, rotation_costs) = tables

    def centre(cell: Cell) -> Point:
        return _cell_to_world(cell, resolution, origin_x, origin_y)

    raw_free_flat = raw_free_mask.reshape(-1)
    allowed_center_flat = allowed_center_mask.reshape(-1)
    penalty_flat = (
        penalty_mask.reshape(-1) if penalty_mask is not None else None)
    clearance_certificate = (
        rotation_safe_mask
        if rotation_safe_mask is not None else turn_safe_mask)
    clearance_certificate_flat = clearance_certificate.reshape(-1)

    def stencil_is_free(row: int, col: int, stencil: Stencil) -> bool:
        flat_offsets, min_row, max_row, min_col, max_col = stencil
        if (row + min_row < 0 or row + max_row >= height
                or col + min_col < 0 or col + max_col >= width):
            return False
        return bool(raw_free_flat[row * width + col + flat_offsets].all())

    def pose_ok(row: int, col: int, heading: int) -> bool:
        key = (row, col, heading)
        cached = pose_cache.get(key)
        if cached is not None:
            return cached
        ok = (
            roi_r0 <= row <= roi_r1 and roi_c0 <= col <= roi_c1
            and bool(allowed_center_mask[row, col])
            and stencil_is_free(row, col, pose_stencils[heading])
        )
        pose_cache[key] = ok
        return ok

    def rotation_ok(row: int, col: int, heading: int, delta: int) -> bool:
        action_code = 1 if delta > 0 else -1
        key = (row, col, heading, action_code)
        cached = primitive_cache.get(key)
        if cached is not None:
            return cached
        # A radial body-clearance certificate proves every intermediate
        # heading at once.  It is intentionally independent of the mission
        # polygon; ``allowed_center_mask`` already enforces route scope.  Only
        # Narrow aligned poses can use exact-body rotation primitives.  The
        # search state below also bounds consecutive same-cell rotation to a
        # commandable shortest turn; this stencil check alone is not enough
        # because a chain of safe 22.5 degree steps could otherwise accumulate
        # into an orientation change greater than 180 degrees.
        ok = bool(
            rotation_safe_mask is not None
            and rotation_safe_mask[row, col])
        if not ok:
            ok = stencil_is_free(
                row, col, rotation_stencils[(heading, delta)])
        primitive_cache[key] = ok
        return ok

    def forward_ok(row: int, col: int, heading: int) -> bool:
        key = (row, col, heading, 2)
        cached = primitive_cache.get(key)
        if cached is not None:
            return cached
        dr, dc = directions[heading]
        nr, nc = row + dr, col + dc
        if not (roi_r0 <= nr <= roi_r1 and roi_c0 <= nc <= roi_c1):
            primitive_cache[key] = False
            return False
        centre_flat_offsets = move_center_flat_offsets[heading]
        base_index = row * width + col
        # The current and destination centres are already inside the convex
        # rectangular ROI/map bounds, therefore every rounded intermediate
        # offset lies inside too; no per-expansion min/max reduction is needed.
        if not allowed_center_flat[
                base_index + centre_flat_offsets].all():
            primitive_cache[key] = False
            return False
        # A line of centres inside the full-turn-clearance mask guarantees
        # every point of the asymmetric body sweep is free.  This is the
        # common room/corridor case and is strictly more conservative than the
        # exact stencil below.  Keep the stencil for narrow, heading-aligned
        # doors where only the 0.68 m body width fits.
        if clearance_certificate_flat[
                base_index + centre_flat_offsets].all():
            primitive_cache[key] = True
            return True
        # Do not apply the usual 2-D "both orthogonal neighbours must be free"
        # corner rule here.  A real diagonal doorway can have both orthogonal
        # cell centres outside the eroded mask while the TD25A rectangle still
        # fits when aligned with the doorway.  The swept footprint stencil
        # below is the authoritative collision check for this SE(2) primitive.
        ok = stencil_is_free(row, col, move_stencils[heading])
        primitive_cache[key] = ok
        return ok

    if goal_yaw is None:
        goal_headings = set(range(heading_count))
    else:
        goal_heading = min(
            range(heading_count),
            key=lambda heading: abs(_wrap_angle(
                goal_yaw - heading_yaws[heading])),
        )
        goal_headings = {goal_heading}
    if start_yaw is None:
        start_headings = range(heading_count)
    else:
        start_headings = [min(
            range(heading_count),
            key=lambda heading: abs(_wrap_angle(
                start_yaw - heading_yaws[heading])),
        )]
    # Track consecutive same-cell rotation steps alongside each best state.
    # It is reset by translation (or inside a radial all-heading certificate).
    # Outside that open core, a run may not exceed a strictly-shorter-than-pi
    # turn because point-path export can command only the wrapped shortest yaw.
    # Keeping this as state metadata rather than a fourth A* dimension avoids a
    # 6-8x expansion blow-up on the Jetson's large maps.
    State = Tuple[int, int, int]
    start_states: List[State] = [
        (start_cell[0], start_cell[1], heading)
        for heading in start_headings
        if pose_ok(start_cell[0], start_cell[1], heading)
    ]
    if not start_states:
        return []

    def heuristic(row: int, col: int) -> float:
        return resolution * math.hypot(row - goal_cell[0], col - goal_cell[1])

    queue: List[Tuple[float, float, State]] = []
    costs: Dict[State, float] = {}
    parents: Dict[State, Optional[State]] = {}
    rotation_runs: Dict[State, int] = {}
    for state in start_states:
        costs[state] = 0.0
        parents[state] = None
        rotation_runs[state] = 0
        heapq.heappush(queue, (heuristic(state[0], state[1]), 0.0, state))
    closed = set()
    final_state: Optional[State] = None

    def rotation_run_angle(end_heading: int, signed_steps: int) -> float:
        if signed_steps == 0:
            return 0.0
        direction = 1 if signed_steps > 0 else -1
        start_heading = (end_heading - signed_steps) % heading_count
        total = 0.0
        cursor_heading = start_heading
        for _ in range(abs(signed_steps)):
            next_heading = (cursor_heading + direction) % heading_count
            total += abs(_wrap_angle(
                heading_yaws[next_heading] - heading_yaws[cursor_heading]))
            cursor_heading = next_heading
        return total

    expansions = 0
    while queue and expansions < max_expansions:
        _, popped_cost, state = heapq.heappop(queue)
        if state in closed or popped_cost > costs.get(state, float("inf")) + 1e-9:
            continue
        row, col, heading = state
        rotation_run = rotation_runs.get(state, 0)
        if (row, col) == goal_cell and heading in goal_headings:
            final_state = state
            break
        closed.add(state)
        expansions += 1
        neighbours: List[Tuple[State, float, int]] = []
        for delta in (-1, 1):
            next_heading = (heading + delta) % heading_count
            if rotation_ok(row, col, heading, delta):
                radial_rotation_safe = bool(
                    rotation_safe_mask is not None
                    and rotation_safe_mask[row, col])
                if radial_rotation_safe:
                    next_rotation_run = 0
                else:
                    next_rotation_run = (
                        rotation_run + delta
                        if rotation_run == 0
                        or (rotation_run > 0) == (delta > 0)
                        else delta
                    )
                    if rotation_run_angle(
                            next_heading, next_rotation_run) >= math.pi - 1e-6:
                        continue
                neighbours.append((
                    (row, col, next_heading),
                    rotation_costs[(heading, delta)],
                    next_rotation_run,
                ))
        if forward_ok(row, col, heading):
            dr, dc = directions[heading]
            move_cost = resolution * math.hypot(dr, dc)
            if penalty_flat is not None:
                centre_flat_offsets = move_center_flat_offsets[heading]
                base_index = row * width + col
                if penalty_flat[base_index + centre_flat_offsets].any():
                    move_cost *= 25.0
            neighbours.append((
                (row + dr, col + dc, heading),
                move_cost,
                0,
            ))
        for next_state, step_cost, next_rotation_run in neighbours:
            tentative = popped_cost + step_cost
            if tentative + 1e-9 >= costs.get(next_state, float("inf")):
                continue
            costs[next_state] = tentative
            parents[next_state] = state
            rotation_runs[next_state] = next_rotation_run
            heapq.heappush(queue, (
                tentative + heuristic(next_state[0], next_state[1]),
                tentative, next_state,
            ))
    if final_state is None:
        return []
    states: List[State] = []
    cursor: Optional[State] = final_state
    while cursor is not None:
        states.append(cursor)
        cursor = parents[cursor]
    states.reverse()
    points: List[Point] = [start]
    last_cell: Optional[Cell] = None
    for row, col, _ in states:
        cell = (row, col)
        if cell == last_cell:
            continue
        _extend_dedup(points, [centre(cell)])
        last_cell = cell
    _extend_dedup(points, [goal])
    points = _simplify_collinear(points)
    points = _densify_polyline(points, path_step_m)
    valid, _ = _validate_connector_candidate_polyline(
        raw_free_mask, points, resolution, origin_x, origin_y,
        footprint=footprint)
    return points if valid else []


def _connect_points_footprint_safe(
    allowed_center_mask: np.ndarray,
    turn_safe_mask: np.ndarray,
    raw_free_mask: np.ndarray,
    start: Point,
    goal: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    footprint: CoverageFootprint,
    start_yaw: Optional[float] = None,
    goal_yaw: Optional[float] = None,
    max_se2_expansions: int = 180_000,
    rotation_safe_mask: Optional[np.ndarray] = None,
    stencil_table_cache: Optional[
        Dict[Tuple[object, ...], object]
    ] = None,
    penalty_mask: Optional[np.ndarray] = None,
) -> List[Point]:
    """Prefer cheap 2-D A*, then repair unsafe connectors in SE(2)."""
    if _dist(start, goal) <= 1e-6:
        return [start]
    ordinary_penalty = ~turn_safe_mask
    if penalty_mask is not None:
        ordinary_penalty |= penalty_mask
    ordinary = _connect_points(
        allowed_center_mask, start, goal,
        resolution, origin_x, origin_y, path_step_m,
        penalty_mask=ordinary_penalty, prefer_theta=False)
    orientation_constrained = start_yaw is not None or goal_yaw is not None
    if ordinary and not orientation_constrained:
        valid, _ = _validate_connector_candidate_polyline(
            raw_free_mask, ordinary, resolution, origin_x, origin_y,
            footprint=footprint)
        if valid:
            return ordinary
        # Most indoor failures are not arbitrary SE(2) mazes: a room-safe path
        # crosses one short doorway throat.  Replace each non-turn-safe run by a
        # single straight, body-validated portal traversal, while doing every
        # heading change at least 0.60m inside the room core.  This is both more
        # predictable and much cheaper than launching a lattice search first.
        flags = [
            _point_is_free(
                turn_safe_mask, point, resolution, origin_x, origin_y)
            for point in ordinary
        ]
        unsafe_runs: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        for index, safe in enumerate(flags + [True]):
            if not safe and run_start is None:
                run_start = index
            elif safe and run_start is not None:
                unsafe_runs.append((run_start, index - 1))
                run_start = None
        if unsafe_runs and flags[0] and flags[-1]:
            repaired: List[Point] = [start]
            cursor_index = 0
            repair_ok = True
            for unsafe_start, unsafe_end in unsafe_runs:
                entry_index = max(cursor_index, unsafe_start - 1)
                walked = 0.0
                while entry_index > cursor_index and walked < 0.65:
                    walked += _dist(
                        ordinary[entry_index - 1], ordinary[entry_index])
                    entry_index -= 1
                exit_index = min(len(ordinary) - 1, unsafe_end + 1)
                walked = 0.0
                while exit_index < len(ordinary) - 1 and walked < 0.65:
                    walked += _dist(
                        ordinary[exit_index], ordinary[exit_index + 1])
                    exit_index += 1
                safe_leg = _connect_points(
                    turn_safe_mask, repaired[-1], ordinary[entry_index],
                    resolution, origin_x, origin_y, path_step_m,
                    prefer_theta=False)
                portal = _densify(
                    ordinary[entry_index], ordinary[exit_index], path_step_m)
                portal_valid, _ = _validate_connector_candidate_polyline(
                    raw_free_mask, portal, resolution, origin_x, origin_y,
                    footprint=footprint)
                if (not safe_leg or not portal_valid
                        or not _world_segment_is_free(
                            allowed_center_mask,
                            ordinary[entry_index], ordinary[exit_index],
                            resolution, origin_x, origin_y)):
                    repair_ok = False
                    break
                _extend_dedup(repaired, safe_leg[1:])
                _extend_dedup(repaired, portal[1:])
                cursor_index = exit_index
            if repair_ok:
                final_leg = _connect_points(
                    turn_safe_mask, repaired[-1], goal,
                    resolution, origin_x, origin_y, path_step_m,
                    prefer_theta=False)
                if final_leg:
                    _extend_dedup(repaired, final_leg[1:])
                    repaired_valid, _ = _validate_connector_candidate_polyline(
                        raw_free_mask, repaired, resolution, origin_x, origin_y,
                        footprint=footprint)
                    if repaired_valid:
                        return repaired
    return _astar_se2_lattice(
        raw_free_mask=raw_free_mask,
        allowed_center_mask=allowed_center_mask,
        turn_safe_mask=turn_safe_mask,
        rotation_safe_mask=rotation_safe_mask,
        start=start,
        goal=goal,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        footprint=footprint,
        path_step_m=path_step_m,
        start_yaw=start_yaw,
        goal_yaw=goal_yaw,
        guide_path=ordinary,
        max_expansions=max(1, int(max_se2_expansions)),
        stencil_table_cache=stencil_table_cache,
        penalty_mask=penalty_mask,
    )


def _connect_points_orthogonal_footprint_safe(
    allowed_center_mask: np.ndarray,
    raw_free_mask: np.ndarray,
    start: Point,
    goal: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    footprint: CoverageFootprint,
    penalty_mask: Optional[np.ndarray] = None,
) -> List[Point]:
    """Connect room-cleaning phases without a long diagonal shortcut.

    The indoor map and its BCD cells are axis aligned.  A collision-free
    diagonal between completed cells can nevertheless cross many parallel
    cleaning lanes.  Try both one-corner Manhattan paths, then four-neighbour
    A*.  The exact TD25A rectangle is validated before accepting the route.
    """
    if _dist(start, goal) <= 1e-6:
        return [start]

    candidates: List[List[Point]] = []
    if (abs(start[0] - goal[0]) <= resolution * 0.25
            or abs(start[1] - goal[1]) <= resolution * 0.25):
        candidates.append([start, goal])
    candidates.extend((
        [start, (goal[0], start[1]), goal],
        [start, (start[0], goal[1]), goal],
    ))
    for sparse in candidates:
        compact = _simplify_collinear(sparse)
        if not _polyline_is_free(
                allowed_center_mask, compact,
                resolution, origin_x, origin_y):
            continue
        if (penalty_mask is not None
                and not _polyline_is_free(
                    ~penalty_mask, compact,
                    resolution, origin_x, origin_y)):
            continue
        dense = _densify_polyline(compact, path_step_m)
        valid, _ = _validate_connector_candidate_polyline(
            raw_free_mask, dense,
            resolution, origin_x, origin_y,
            footprint=footprint)
        continuous_valid, _ = validate_footprint_path(
            raw_free_mask, dense,
            resolution, origin_x, origin_y,
            footprint=footprint, lookahead_m=0.0)
        if valid and continuous_valid:
            return dense

    start_cell = _snap_to_free(
        allowed_center_mask,
        _world_to_cell(start, resolution, origin_x, origin_y))
    goal_cell = _snap_to_free(
        allowed_center_mask,
        _world_to_cell(goal, resolution, origin_x, origin_y))
    if start_cell is None or goal_cell is None:
        return []
    queue: List[Tuple[float, Cell]] = [(0.0, start_cell)]
    parents: Dict[Cell, Optional[Cell]] = {start_cell: None}
    costs: Dict[Cell, float] = {start_cell: 0.0}

    def heuristic(cell: Cell) -> float:
        return abs(cell[0] - goal_cell[0]) + abs(cell[1] - goal_cell[1])

    while queue:
        _, cell = heapq.heappop(queue)
        if cell == goal_cell:
            cell_path: List[Cell] = []
            cursor: Optional[Cell] = cell
            while cursor is not None:
                cell_path.append(cursor)
                cursor = parents[cursor]
            cell_path.reverse()
            points = [
                _cell_to_world(item, resolution, origin_x, origin_y)
                for item in cell_path
            ]
            points[0] = start
            points[-1] = goal
            points = _simplify_collinear(points)
            points = _densify_polyline(points, path_step_m)
            valid, _ = _validate_connector_candidate_polyline(
                raw_free_mask, points,
                resolution, origin_x, origin_y,
                footprint=footprint)
            continuous_valid, _ = validate_footprint_path(
                raw_free_mask, points,
                resolution, origin_x, origin_y,
                footprint=footprint, lookahead_m=0.0)
            return points if valid and continuous_valid else []
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            next_cell = (cell[0] + dy, cell[1] + dx)
            if not _cell_is_free(allowed_center_mask, next_cell):
                continue
            crossing_penalty = (
                24.0 if penalty_mask is not None
                and bool(penalty_mask[next_cell]) else 0.0)
            next_cost = costs[cell] + 1.0 + crossing_penalty
            if next_cost + 1e-9 >= costs.get(next_cell, float("inf")):
                continue
            costs[next_cell] = next_cost
            parents[next_cell] = cell
            heapq.heappush(
                queue, (next_cost + heuristic(next_cell), next_cell))
    return []


def _rotate_ring_near_start(
    ring: Sequence[Point],
    start: Point,
    path_step_m: float,
    safe_start_mask: Optional[np.ndarray] = None,
    resolution: float = 0.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> List[Point]:
    polyline = list(ring)
    if len(polyline) < 2:
        return []
    closed = _dist(polyline[0], polyline[-1]) <= path_step_m * 2.5
    core = polyline[:-1] if closed else polyline
    if not core:
        return []
    if closed:
        eligible = list(range(len(core)))
        if safe_start_mask is not None and resolution > 0.0:
            safe_indices = [
                index for index, point in enumerate(core)
                if _point_is_free(
                    safe_start_mask, point,
                    resolution, origin_x, origin_y)
            ]
            if safe_indices:
                eligible = safe_indices
        nearest = min(eligible, key=lambda idx: _dist(start, core[idx]))
        rotated = core[nearest:] + core[:nearest]
        rotated.append(rotated[0])
        return rotated
    if _dist(start, core[-1]) < _dist(start, core[0]):
        core.reverse()
    return core


def _ring_start_variants(
    ring: Sequence[Point],
    start: Point,
    path_step_m: float,
    safe_start_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    max_variants: int = 8,
) -> List[List[Point]]:
    """Return separated, safe closed-ring starts including axis alignments."""
    polyline = list(ring)
    if len(polyline) < 2:
        return []
    closed = _dist(polyline[0], polyline[-1]) <= path_step_m * 2.5
    if not closed:
        rotated = _rotate_ring_near_start(
            polyline, start, path_step_m,
            safe_start_mask=safe_start_mask,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y)
        return [rotated] if rotated else []
    core = polyline[:-1]
    eligible = [
        index for index, point in enumerate(core)
        if _point_is_free(
            safe_start_mask, point,
            resolution, origin_x, origin_y)
    ]
    if not eligible:
        eligible = list(range(len(core)))
    # A slightly farther start sharing x or y with the current point often
    # gives one straight connector, whereas the Euclidean-nearest point needs a
    # shallow diagonal whose 0.55 m execution yaw clips a doorway corner.
    ranked = sorted(
        eligible,
        key=lambda index: (
            3.0 * min(
                abs(core[index][0] - start[0]),
                abs(core[index][1] - start[1]))
            + _dist(start, core[index]),
            _dist(start, core[index]),
        ),
    )
    chosen_indices: List[int] = []
    separation_m = max(0.40, path_step_m * 4.0)
    for index in ranked:
        if any(_dist(core[index], core[other]) < separation_m
               for other in chosen_indices):
            continue
        chosen_indices.append(index)
        if len(chosen_indices) >= max(1, int(max_variants)):
            break
    variants: List[List[Point]] = []
    for index in chosen_indices:
        rotated = core[index:] + core[:index]
        rotated.append(rotated[0])
        variants.append(rotated)
    return variants


def _polyline_area(polyline: Sequence[Point], close_tolerance_m: float) -> float:
    core = list(polyline)
    if len(core) > 1 and _dist(core[0], core[-1]) <= close_tolerance_m:
        core = core[:-1]
    if len(core) < 3:
        return 0.0
    return 0.5 * abs(sum(
        core[index][0] * core[(index + 1) % len(core)][1]
        - core[(index + 1) % len(core)][0] * core[index][1]
        for index in range(len(core))))


def _rounded_rectangle_path(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    corner_radius_m: float,
    path_step_m: float,
) -> List[Point]:
    """Clockwise-free rounded rectangle with tangent-continuous corners."""
    radius = min(
        float(corner_radius_m),
        0.5 * (x_max - x_min),
        0.5 * (y_max - y_min),
    )
    if radius <= 0.0 or x_max <= x_min or y_max <= y_min:
        return []
    step = max(0.02, float(path_step_m))
    path: List[Point] = []

    def append_line(start: Point, end: Point) -> None:
        _extend_dedup(path, _densify(start, end, step))

    def append_arc(center: Point, start_angle: float, end_angle: float) -> None:
        count = max(
            8,
            int(math.ceil(abs(end_angle - start_angle) * radius / step)),
        )
        points = [(
            center[0] + radius * math.cos(
                start_angle + (end_angle - start_angle) * index / count),
            center[1] + radius * math.sin(
                start_angle + (end_angle - start_angle) * index / count),
        ) for index in range(count + 1)]
        _extend_dedup(path, points)

    append_line((x_min + radius, y_min), (x_max - radius, y_min))
    append_arc((x_max - radius, y_min + radius), -math.pi / 2.0, 0.0)
    append_line((x_max, y_min + radius), (x_max, y_max - radius))
    append_arc((x_max - radius, y_max - radius), 0.0, math.pi / 2.0)
    append_line((x_max - radius, y_max), (x_min + radius, y_max))
    append_arc((x_min + radius, y_max - radius), math.pi / 2.0, math.pi)
    append_line((x_min, y_max - radius), (x_min, y_min + radius))
    append_arc((x_min + radius, y_min + radius), math.pi, 3.0 * math.pi / 2.0)
    return path


def _plan_near_rectangular_perimeter(
    region_travel_mask: np.ndarray,
    raw_free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    footprint: CoverageFootprint,
    clean_width_m: float,
) -> List[Point]:
    """Plan one close, rounded red loop for a rectangular/near-rect cell.

    A circular erosion forces the long TD25A rear corner to stay 0.87 m from
    every straight wall.  A tangent-continuous rounded rectangle is less
    conservative while still validated with the execution-side 0.55 m yaw
    lookahead and the exact 1.05 m x 0.68 m body sweep.
    """
    geometry = _mask_geometry(region_travel_mask)
    if geometry is None or geometry[2] < 0.90:
        return []
    _, (row_min, row_max, col_min, col_max), _, _ = geometry
    base_x_min = origin_x + (col_min + 0.5) * resolution
    base_x_max = origin_x + (col_max + 0.5) * resolution
    base_y_min = origin_y + (row_min + 0.5) * resolution
    base_y_max = origin_y + (row_max + 0.5) * resolution
    body = footprint
    # Partition cuts are artificial and adjacent cells can differ by a thin
    # raster seam.  Permit the rounded loop to borrow at most 0.25 m across
    # that seam; the exact raw-map body sweep below still forbids walls and the
    # route never leaves the physical room through this tolerance.
    region_path_mask = dilate_binary(
        region_travel_mask,
        max(1, int(math.ceil(0.25 / resolution))))
    best: List[Point] = []
    best_key = (-1, float("-inf"), float("-inf"))
    # The travel mask is already about half a body width from physical walls.
    # These additional insets put the straight edge pass at roughly 0.59--0.69m
    # wall clearance; the cleaning disk still reaches the 0.34m travel edge.
    for inset_m in (
            0.25, 0.30, 0.35, 0.20, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        x_min = base_x_min + inset_m
        x_max = base_x_max - inset_m
        y_min = base_y_min + inset_m
        y_max = base_y_max - inset_m
        if x_max - x_min < 1.50 or y_max - y_min < 1.50:
            continue
        for radius_m in (0.85, 1.00, 1.20, 0.75, 1.35, 1.50):
            if (x_max - x_min < 2.0 * radius_m
                    or y_max - y_min < 2.0 * radius_m):
                continue
            candidate = _rounded_rectangle_path(
                x_min, x_max, y_min, y_max,
                radius_m, path_step_m)
            if len(candidate) < 3:
                continue
            if not _polyline_is_free(
                    region_path_mask, candidate,
                    resolution, origin_x, origin_y):
                continue
            valid, violations = validate_footprint_path(
                raw_free_mask, candidate,
                resolution, origin_x, origin_y,
                footprint=body, lookahead_m=0.0)
            if not valid or violations:
                continue
            cleaned = _polyline_cleaning_mask(
                region_travel_mask.shape, candidate,
                resolution, origin_x, origin_y, clean_width_m)
            covered = int((cleaned & region_travel_mask).sum())
            # Maximise real edge coverage, then prefer fewer metres and a larger
            # radius (gentler MPPI curvature) when coverage is tied.
            length_m = sum(
                _dist(first, second)
                for first, second in zip(candidate, candidate[1:]))
            key = (covered, -length_m, radius_m)
            if key > best_key:
                best_key = key
                best = candidate
    return best


def _elasticize_perimeter_ring(
    source_ring: Sequence[Point],
    safe_component_mask: np.ndarray,
    room_travel_mask: np.ndarray,
    global_travel_mask: np.ndarray,
    global_turn_safe_mask: np.ndarray,
    raw_free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    footprint: CoverageFootprint,
    clean_width_m: float,
) -> List[Point]:
    """Pull straight parts of one safe boundary ring closer to real walls.

    The ring stays unchanged through corners and around one explicit safe entry
    anchor.  Every selected candidate passes the controller's continuous
    immediate-tangent body sweep; the anchor itself remains in full-turn-safe
    space for the phase handoff.  It changes one coherent loop only and
    therefore cannot introduce an extra crossing or a disconnected fragment.
    """
    ring = _rdp_masked(
        source_ring, max(0.10, resolution * 1.25),
        safe_component_mask, resolution, origin_x, origin_y)
    ring = _densify_polyline(ring, path_step_m)
    if len(ring) < 8:
        return []
    if _dist(ring[0], ring[-1]) > path_step_m * 2.5:
        ring.append(ring[0])
    core = ring[:-1]
    count = len(core)
    if count < 6:
        return []

    body_samples = _footprint_samples(footprint, resolution)
    look = max(2, int(math.ceil(0.35 / max(0.04, path_step_m))))
    max_pull_m = max(0.0, footprint.turn_clearance_m - 0.43)
    pull_steps = max(
        1, int(math.ceil(max_pull_m / max(0.04, resolution * 0.50))))
    physical_boundary = global_travel_mask & dilate_binary(
        ~global_travel_mask, 1)
    boundary_near = dilate_binary(
        physical_boundary,
        max(2, int(math.ceil(0.55 / resolution))))
    height, width = room_travel_mask.shape
    targets: List[Point] = []
    raw_alpha: List[float] = []
    for index, point in enumerate(core):
        before = core[(index - look) % count]
        after = core[(index + look) % count]
        tangent_x = after[0] - before[0]
        tangent_y = after[1] - before[1]
        tangent_norm = math.hypot(tangent_x, tangent_y)
        if tangent_norm <= 1e-6:
            targets.append(point)
            raw_alpha.append(0.0)
            continue
        tangent_x /= tangent_norm
        tangent_y /= tangent_norm
        yaw = math.atan2(tangent_y, tangent_x)
        best_target = point
        best_distance = 0.0
        for normal_x, normal_y in (
                (-tangent_y, tangent_x), (tangent_y, -tangent_x)):
            candidate_target = point
            candidate_distance = 0.0
            escaped_turn_safe = False
            for step_index in range(1, pull_steps + 1):
                distance_m = max_pull_m * step_index / pull_steps
                candidate = (
                    point[0] + normal_x * distance_m,
                    point[1] + normal_y * distance_m,
                )
                row, col = _world_to_cell(
                    candidate, resolution, origin_x, origin_y)
                if not (0 <= row < height and 0 <= col < width
                        and room_travel_mask[row, col]
                        and boundary_near[row, col]):
                    break
                if not _footprint_pose_is_free(
                        raw_free_mask, candidate, yaw, body_samples,
                        resolution, origin_x, origin_y):
                    break
                escaped_turn_safe = (
                    escaped_turn_safe
                    or not global_turn_safe_mask[row, col])
                candidate_target = candidate
                candidate_distance = distance_m
            if escaped_turn_safe and candidate_distance > best_distance:
                best_target = candidate_target
                best_distance = candidate_distance

        previous_heading = math.atan2(
            point[1] - before[1], point[0] - before[0])
        next_heading = math.atan2(
            after[1] - point[1], after[0] - point[0])
        bend = abs(_wrap_angle(next_heading - previous_heading))
        if bend <= math.radians(6.0):
            straight_weight = 1.0
        elif bend >= math.radians(24.0):
            straight_weight = 0.0
        else:
            straight_weight = (
                math.radians(24.0) - bend) / math.radians(18.0)
        targets.append(best_target)
        raw_alpha.append(
            straight_weight if best_distance > 0.02 else 0.0)

    shoulder = max(
        2, int(math.ceil(0.75 / max(0.04, path_step_m))))
    alpha = list(raw_alpha)
    for index in range(count):
        corner_strength = 1.0 - raw_alpha[index]
        if corner_strength <= 1e-6:
            continue
        for delta in range(-shoulder, shoulder + 1):
            influence = corner_strength * (
                1.0 - abs(delta) / (shoulder + 1))
            target_index = (index + delta) % count
            alpha[target_index] = min(
                alpha[target_index], 1.0 - influence)

    # Preserve a short, full-turn-clearance entry window even on a long nearly
    # straight obstacle ring.  The preceding transfer can then stop and align
    # without borrowing clearance from the close wall pass.
    anchor = min(range(count), key=lambda index: alpha[index])
    anchor_shoulder = max(
        1, int(math.ceil(0.25 / max(0.04, path_step_m))))
    for delta in range(-anchor_shoulder, anchor_shoulder + 1):
        alpha[(anchor + delta) % count] = 0.0

    # Pull is monotonic toward the physical wall.  The first valid (largest)
    # scale is therefore the intended close-wall pass; evaluating every weaker
    # scale and rescoring the entire map was a large ARM/Jetson cost with no
    # safety benefit.
    for scale in (1.0, 0.80, 0.60, 0.40, 0.20, 0.0):
        candidate = [(
            point[0] + (target[0] - point[0]) * weight * scale,
            point[1] + (target[1] - point[1]) * weight * scale,
        ) for point, target, weight in zip(core, targets, alpha)]
        candidate.append(candidate[0])
        candidate = _densify_polyline(candidate, path_step_m)
        if not _polyline_is_free(
                room_travel_mask, candidate,
                resolution, origin_x, origin_y):
            continue
        continuous_valid, continuous_violations = validate_footprint_path(
            raw_free_mask, candidate,
            resolution, origin_x, origin_y,
            footprint=footprint, lookahead_m=0.0)
        if not continuous_valid or continuous_violations:
            continue
        return candidate
    return []


def _extract_wall_follow_arcs(
    room_travel_mask: np.ndarray,
    global_travel_mask: np.ndarray,
    raw_free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    footprint: CoverageFootprint,
    min_segment_m: float = 0.60,
) -> List[List[Point]]:
    """Return exact-footprint-safe red arcs beside physical room walls.

    The half-width travel mask lets an aligned rectangular TD25A clean much
    closer to a straight wall than the circumscribed turn-safe disk.  Artificial
    room-partition cuts are excluded by retaining only edges that are also on
    the boundary of the global travel component.  Unsafe corner rotations split
    the wall trace into separate arcs; blue connectors can then turn inside the
    room before approaching the next red arc.
    """
    rings = _trace_perimeter_rings(
        room_travel_mask, resolution, origin_x, origin_y,
        step_m=path_step_m, min_ring_cells=8)
    if not rings:
        return []
    ring = max(
        rings,
        key=lambda candidate: _polyline_area(
            candidate, path_step_m * 2.5))
    if len(ring) < 2:
        return []
    simplified = _rdp_masked(
        ring, max(0.18, resolution * 2.0),
        room_travel_mask, resolution, origin_x, origin_y)
    if len(simplified) < 2:
        return []
    global_boundary = global_travel_mask & dilate_binary(
        ~global_travel_mask, 1)
    height, width = room_travel_mask.shape
    arcs: List[List[Point]] = []
    current: List[Point] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            dense = _densify_polyline(current, path_step_m)
            if sum(_dist(a, b) for a, b in zip(dense, dense[1:])) >= min_segment_m:
                arcs.append(dense)
        current = []

    for start, end in zip(simplified, simplified[1:]):
        midpoint = (
            0.5 * (start[0] + end[0]),
            0.5 * (start[1] + end[1]),
        )
        row, col = _world_to_cell(
            midpoint, resolution, origin_x, origin_y)
        candidate_length = _dist(start, end)
        physical_boundary = (
            0 <= row < height and 0 <= col < width
            and bool(global_boundary[row, col])
        )
        segment_valid = False
        if physical_boundary and candidate_length >= min_segment_m:
            segment_valid, _ = _validate_connector_candidate_polyline(
                raw_free_mask, [start, end],
                resolution, origin_x, origin_y, footprint=footprint)
        if not segment_valid:
            flush()
            continue
        if not current:
            current = [start, end]
            continue
        if _dist(current[-1], start) > max(0.02, resolution * 0.35):
            flush()
            current = [start, end]
            continue
        trial = current + [end]
        trial_valid, _ = _validate_connector_candidate_polyline(
            raw_free_mask, trial,
            resolution, origin_x, origin_y, footprint=footprint)
        if trial_valid:
            current = trial
        else:
            flush()
            current = [start, end]
    flush()
    return arcs


def _prepare_forward_wall_arc(
    source_arc: Sequence[Point],
    travel_mask: np.ndarray,
    turn_safe_mask: np.ndarray,
    raw_free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    footprint: CoverageFootprint,
) -> Tuple[List[Point], List[Point]]:
    """Trim a close-wall pass until TD25A can leave it without reversing.

    A wall-tangent pose is safe with only half the body width of clearance, but
    an in-place turn there is not.  The long rear overhang also makes reversing
    the same arc invalid.  Preserve the validated forward direction, shorten
    its terminal end in 0.25 m increments, then use a tangent-continuous
    quarter-circle to move inward while the heading changes.  Returning an
    empty pair rejects a dead-end wall pass rather than leaving execution
    stranded at a corner.
    """
    source = _densify_polyline(source_arc, path_step_m)
    initial_yaw = _initial_path_yaw(source)
    if len(source) < 2 or initial_yaw is None:
        return [], []
    # Moving exactly on the half-width travel boundary leaves no clearance for
    # the rear corner's initial outward swing.  A 5--30 cm inward offset still
    # cleans substantially closer than the ordinary turn-safe contour while
    # making a forward curve physically possible.
    left_normal = (-math.sin(initial_yaw), math.cos(initial_yaw))
    curve_options = (
        (4.0, math.radians(25.0)),
        (5.0, math.radians(20.0)),
        (3.0, math.radians(30.0)),
        (2.5, math.radians(35.0)),
        (2.0, math.radians(40.0)),
        (1.6, math.radians(50.0)),
        (1.2, math.radians(65.0)),
        (0.9, math.radians(90.0)),
        (6.0, math.radians(15.0)),
    )
    for offset_m in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        for inward_sign in (1.0, -1.0):
            arc = [(
                point[0] + inward_sign * offset_m * left_normal[0],
                point[1] + inward_sign * offset_m * left_normal[1],
            ) for point in source]
            straight_valid, _ = _validate_connector_candidate_polyline(
                raw_free_mask, arc,
                resolution, origin_x, origin_y,
                footprint=footprint)
            if not straight_valid:
                continue
            cumulative = [0.0]
            for first, second in zip(arc, arc[1:]):
                cumulative.append(cumulative[-1] + _dist(first, second))
            total_length = cumulative[-1]
            if total_length < 0.60:
                continue
            trim_count = int(math.floor(
                max(0.0, total_length - 0.60) / 0.25))
            for trim_index in range(trim_count + 1):
                retained_length = total_length - trim_index * 0.25
                end_index = max(
                    1,
                    max(index for index, distance in enumerate(cumulative)
                        if distance <= retained_length + 1e-9),
                )
                candidate = arc[:end_index + 1]
                terminal_yaw = _terminal_path_yaw(candidate)
                if terminal_yaw is None:
                    continue
                endpoint = candidate[-1]
                terminal_left = (
                    -math.sin(terminal_yaw), math.cos(terminal_yaw))
                for radius_m, exit_angle in curve_options:
                    sample_count = max(
                        8,
                        int(math.ceil(radius_m * exit_angle
                                      / max(0.04, path_step_m))),
                    )
                    centre = (
                        endpoint[0] + inward_sign * radius_m
                        * terminal_left[0],
                        endpoint[1] + inward_sign * radius_m
                        * terminal_left[1],
                    )
                    radial_angle = math.atan2(
                        endpoint[1] - centre[1],
                        endpoint[0] - centre[0])
                    curve = [(
                        centre[0] + radius_m * math.cos(
                            radial_angle + inward_sign * exit_angle
                            * index / sample_count),
                        centre[1] + radius_m * math.sin(
                            radial_angle + inward_sign * exit_angle
                            * index / sample_count),
                    ) for index in range(sample_count + 1)]
                    exit_yaw = terminal_yaw + inward_sign * exit_angle
                    for straight_m in (0.0, 0.25, 0.50, 0.75, 1.00):
                        escape = list(curve)
                        if straight_m > 0.0:
                            extension = _densify(
                                escape[-1],
                                (escape[-1][0]
                                 + math.cos(exit_yaw) * straight_m,
                                 escape[-1][1]
                                 + math.sin(exit_yaw) * straight_m),
                                path_step_m)
                            _extend_dedup(escape, extension[1:])
                        if not _point_is_free(
                                turn_safe_mask, escape[-1],
                                resolution, origin_x, origin_y):
                            continue
                        if not _polyline_is_free(
                                travel_mask, escape,
                                resolution, origin_x, origin_y):
                            continue
                        combined = list(candidate)
                        _extend_dedup(combined, escape[1:])
                        valid, violations = validate_footprint_path(
                            raw_free_mask, combined,
                            resolution, origin_x, origin_y,
                            footprint=footprint, lookahead_m=0.0)
                        if valid and not violations:
                            return candidate, escape
    return [], []


def _plan_partitioned_coverage_once(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_world: Point,
    swath_spacing_m: float,
    robot_yaw: Optional[float] = None,
    footprint: Optional[CoverageFootprint] = None,
    clip_polygon: Optional[Sequence[Point]] = None,
    selection_boundary_polygon: Optional[Sequence[Point]] = None,
    blocked_polygons: Optional[Sequence[Sequence[Point]]] = None,
    known_free_mask: Optional[np.ndarray] = None,
    path_step_m: float = 0.10,
    min_swath_m: float = 0.45,
    min_region_area_m2: float = 3.0,
    min_cleanable_component_area_m2: float = 0.80,
    min_useful_swath_m: float = 1.30,
    min_useful_region_area_m2: float = 0.0,
    min_useful_region_lane_m: float = 0.0,
    min_fragment_cell_lane_m: float = 0.0,
    adaptive_fragment_pruning: bool = False,
    max_regions: int = 16,
    clean_width_m: float = 0.70,
    exit_aware_enabled: bool = False,
    region_order_mode: str = "station",
    avoid_completed_route_transfers: bool = False,
    avoid_future_component_swaths: bool = False,
    avoid_fill_to_perimeter_crossings: bool = False,
    hard_avoid_completed_route_transfers: bool = False,
    avoid_future_region_swaths: bool = False,
    refine_inter_region_transfers: bool = False,
    refine_against_all_routes: bool = False,
    refine_transfer_reverse: bool = False,
    refine_same_region_transfers: bool = False,
    refine_fill_only_penalty: bool = False,
    refine_max_extra_turns: int = 0,
    enable_cleaner_semantics: bool = True,
    cleaner_max_offset_m: float = 0.25,
    cleaner_wall_gap_m: float = 0.03,
    cleaner_transition_distance_m: float = 0.45,
) -> PartitionedCoveragePlan:
    """Plan BCD fill by room, then one global physical-boundary perimeter.

    This entry point is intentionally independent from the existing online
    execution state machine.  It returns phase metadata for review images and
    tests; callers must explicitly opt in before any future robot integration.
    """
    body = footprint or CoverageFootprint()
    empty_mask = np.zeros((height, width), dtype=bool)

    def empty(reason: str, free_mask: Optional[np.ndarray] = None,
              snapped: Optional[Point] = None) -> PartitionedCoveragePlan:
        return PartitionedCoveragePlan(
            path=[], segments=[], regions=[], visit_order=[],
            free_mask=empty_mask if free_mask is None else free_mask,
            snapped_start=robot_world if snapped is None else snapped,
            failure_reason=reason, footprint_valid=False,
        )

    if width <= 0 or height <= 0 or resolution <= 0.0:
        return empty("invalid_map_geometry")
    if (min_cleanable_component_area_m2 < 0.0
            or min_useful_swath_m < 0.0
            or min_useful_region_area_m2 < 0.0
            or min_useful_region_lane_m < 0.0
            or min_fragment_cell_lane_m < 0.0):
        return empty("invalid_small_component_threshold")
    if (cleaner_max_offset_m < 0.0 or cleaner_wall_gap_m < 0.0
            or cleaner_transition_distance_m <= 0.0):
        return empty("invalid_cleaner_semantic_parameters")
    grid = np.frombuffer(data, dtype=np.int8).reshape(height, width).copy()
    if known_free_mask is not None:
        if known_free_mask.shape != grid.shape:
            return empty("known_free_shape_mismatch")
        legacy_false_free = (grid >= 0) & (grid <= 50) & ~known_free_mask.astype(bool)
        grid[legacy_false_free] = -1
    effective_data = grid.tobytes()
    mission_mask = (
        polygon_to_mask(
            clip_polygon, width, height, resolution, origin_x, origin_y)
        if clip_polygon is not None else np.ones((height, width), dtype=bool)
    )
    selection_boundary_mask = (
        polygon_to_mask(
            selection_boundary_polygon,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
        )
        if selection_boundary_polygon is not None else None)

    # Keep a global travel domain for the lead-in from the actual robot pose,
    # while the mission-halo domain remains responsible for room decomposition
    # and later inter-room transfers.
    full_travel_free = build_coverage_free_mask(
        data=effective_data, width=width, height=height,
        resolution=resolution, robot_radius_m=body.half_width_m,
        clip_polygon=None, blocked_polygons=blocked_polygons,
        origin_x=origin_x, origin_y=origin_y,
    )
    robot_start_cell = _snap_to_free(
        full_travel_free,
        _world_to_cell(robot_world, resolution, origin_x, origin_y),
        max_radius=max(width, height),
    )
    if robot_start_cell is None:
        return empty("robot_not_in_global_travel_space", full_travel_free)
    global_travel_reachable = largest_component(
        full_travel_free, seed=robot_start_cell)

    mission_travel_free = global_travel_reachable.copy()
    if clip_polygon is not None:
        halo_cells = max(1, int(math.ceil(1.0 / resolution)))
        mission_travel_free &= dilate_binary(mission_mask, halo_cells)
    start_cell = _snap_to_free(
        mission_travel_free,
        robot_start_cell,
        max_radius=max(width, height),
    )
    if start_cell is None:
        return empty("mission_not_reachable_from_robot", mission_travel_free)
    travel_reachable = largest_component(mission_travel_free, seed=start_cell)
    snapped_start = _cell_to_world(
        robot_start_cell, resolution, origin_x, origin_y)

    # All room turns, fill lanes, and red boundary paths use the complete body's
    # circumscribed rear-corner clearance.  Narrow-door travel is the only phase
    # allowed to leave this mask and is checked again with the exact rectangle.
    turn_safe = _build_radial_clearance_mask(
        data=effective_data, width=width, height=height,
        resolution=resolution, clearance_m=body.turn_clearance_m,
        blocked_polygons=blocked_polygons,
        origin_x=origin_x, origin_y=origin_y,
    )
    # Keep the raw radial certificate separate from the cleaning-region mask.
    # Connector scopes already enforce the mission polygon and reachability;
    # using this global copy lets the SE(2) lattice certify body sweeps near a
    # simplified room/door boundary without repeating hundreds of samples.
    rotation_safe_global = turn_safe.copy()
    turn_safe &= travel_reachable & mission_mask
    if not turn_safe.any():
        return empty("no_turn_safe_cleaning_space", travel_reachable, snapped_start)

    room_masks = _partition_travel_component(
        travel_reachable & dilate_binary(mission_mask, 1),
        resolution=resolution,
        min_region_area_m2=min_region_area_m2,
        max_regions=max_regions,
    )
    if not room_masks:
        return empty("room_decomposition_failed", travel_reachable, snapped_start)

    regions: List[CoverageRegion] = []
    cells_by_region: Dict[
        int, List[Tuple[np.ndarray, str, float, List[Swath], Point]]
    ] = {}
    skipped_room_count = 0
    discarded_small_component_count = 0
    discarded_small_turn_safe = np.zeros((height, width), dtype=bool)
    for room_mask in room_masks:
        fill_mask = turn_safe & room_mask
        geometry = _mask_geometry(fill_mask)
        if geometry is None:
            skipped_room_count += 1
            continue
        _, (r0, r1, c0, c1), _, _ = geometry
        # Inspect every turn-safe island.  A tiny, short lidar pocket should not
        # create its own clean/turn/perimeter phase, but a long narrow corridor
        # is still useful even when its centre-mask area is small.  Therefore an
        # island is discarded only when it fails both the area and useful-straight
        # criteria; it remains available for travel in either case.
        fill_components = _connected_components_fast(fill_mask, min_cells=1)
        component_plans: List[
            Tuple[np.ndarray, str, float, List[Swath], Point]
        ] = []
        fragmentation_profiles: List[Dict[str, float]] = []
        accepted_fill_mask = np.zeros((height, width), dtype=bool)
        for component in fill_components:
            component_geometry = _mask_geometry(component)
            if component_geometry is None:
                continue
            _, (cr0, cr1, cc0, cc1), _, _ = component_geometry
            component_axis, component_angle, initial_swaths = (
                _generate_component_swaths(
                    component, (cr0, cr1, cc0, cc1),
                    resolution, origin_x, origin_y,
                    swath_spacing_m, min_swath_m,
                )
            )
            component_area_m2 = float(component.sum()) * resolution ** 2
            longest_initial_swath_m = max(
                (_dist(start, finish)
                 for start, finish in initial_swaths),
                default=0.0,
            )
            if (component_area_m2 < min_cleanable_component_area_m2
                    and longest_initial_swath_m < min_useful_swath_m):
                discarded_small_component_count += 1
                discarded_small_turn_safe |= component
                continue
            if not initial_swaths:
                # The body centre can occupy this island, but it contains no
                # straight cleaning run worth entering.  Exclude it from both
                # the green acceptance target and perimeter generation instead
                # of tracing a dense red loop around a zero-benefit pocket.
                discarded_small_component_count += 1
                discarded_small_turn_safe |= component
                continue
            bcd_cells = (
                _boustrophedon_cells(
                    component, component_axis,
                    min_cell_area_m2=max(0.60, min_region_area_m2 * 0.12),
                    resolution=resolution)
                if component_axis in ("x", "y") else [component]
            )
            provisional_cells: List[
                Tuple[np.ndarray, str, float, List[Swath], Point]
            ] = []
            for bcd_cell in bcd_cells:
                cell_geometry = _mask_geometry(bcd_cell)
                if cell_geometry is None:
                    continue
                _, (br0, br1, bc0, bc1), _, _ = cell_geometry
                if component_axis in ("x", "y"):
                    cell_swaths = _generate_swaths_from_mask(
                        bcd_cell[br0:br1 + 1, bc0:bc1 + 1],
                        resolution,
                        origin_x + bc0 * resolution,
                        origin_y + br0 * resolution,
                        swath_spacing_m=swath_spacing_m,
                        axis=component_axis,
                        min_swath_m=min_swath_m,
                    )
                    cell_axis = component_axis
                    cell_angle = component_angle
                else:
                    cell_axis, cell_angle, cell_swaths = (
                        _generate_component_swaths(
                            bcd_cell, (br0, br1, bc0, bc1),
                            resolution, origin_x, origin_y,
                            swath_spacing_m, min_swath_m,
                        )
                    )
                if not cell_swaths:
                    # BCD often cuts a useful room into a main rectangle plus
                    # thin furniture-side fingers.  A finger without a useful
                    # lane must not inherit the parent component's perimeter;
                    # otherwise one omitted yellow fragment still creates a
                    # tangled red access/loop/exit sequence.
                    discarded_small_component_count += 1
                    discarded_small_turn_safe |= bcd_cell
                    continue
                component_rows, component_cols = np.nonzero(bcd_cell)
                component_center = (
                    origin_x + (float(component_cols.mean()) + 0.5) * resolution,
                    origin_y + (float(component_rows.mean()) + 0.5) * resolution,
                )
                provisional_cells.append((
                    bcd_cell, cell_axis, cell_angle,
                    cell_swaths, component_center))

            adaptive_lane_floor_m = float(min_swath_m)
            provisional_swath_count = sum(
                len(cell[3]) for cell in provisional_cells)
            fragmentation_ratio = (
                len(provisional_cells) / provisional_swath_count
                if provisional_swath_count else 0.0)
            fragmentation_profiles.append({
                "area_m2": float(component_area_m2),
                "cell_count": float(len(provisional_cells)),
                "swath_count": float(provisional_swath_count),
                "ratio": float(fragmentation_ratio),
            })
            if (adaptive_fragment_pruning
                    and len(provisional_cells) >= 2):
                if (component_area_m2 >= 100.0
                        and fragmentation_ratio > 0.30):
                    adaptive_lane_floor_m = max(
                        adaptive_lane_floor_m, 2.0)
                elif (component_area_m2 >= 15.0
                        and fragmentation_ratio > 0.50):
                    adaptive_lane_floor_m = max(
                        adaptive_lane_floor_m, 1.5)
            # Match _generate_swaths_from_mask's cell-count convention: a run
            # of N cells has an endpoint distance of (N-1)*resolution.
            required_lane_cells = max(
                2, int(round(adaptive_lane_floor_m / resolution)))
            required_lane_length_m = (
                (required_lane_cells - 1) * resolution - 1e-8)
            for (bcd_cell, cell_axis, cell_angle,
                 cell_swaths, component_center) in provisional_cells:
                useful_swaths = [
                    swath for swath in cell_swaths
                    if _dist(swath[0], swath[1])
                    >= required_lane_length_m
                ]
                if not useful_swaths:
                    discarded_small_component_count += 1
                    discarded_small_turn_safe |= bcd_cell
                    continue
                useful_lane_m = sum(
                    _dist(swath[0], swath[1]) for swath in useful_swaths)
                if (adaptive_fragment_pruning
                        and component_area_m2 >= 50.0
                        and len(provisional_cells) >= 8
                        and fragmentation_ratio >= 0.30
                        and useful_lane_m < min_fragment_cell_lane_m):
                    # A highly fragmented lidar room can contain many one-lane
                    # BCD chips.  Entering and leaving a chip may cost more than
                    # its entire useful straight run and is a common source of
                    # cross-room knots.  This opt-in floor removes only those
                    # low-yield cells; ordinary rectangular rooms and corridors
                    # are unaffected because their fragmentation ratio is low.
                    discarded_small_component_count += 1
                    discarded_small_turn_safe |= bcd_cell
                    continue
                component_plans.append((
                    bcd_cell, cell_axis, cell_angle,
                    useful_swaths, component_center))
                # Keep only BCD cells that actually contribute a worthwhile
                # straight run.  The prior whole-component promotion retained
                # every narrow side pocket as soon as any sibling cell passed.
                accepted_fill_mask |= bcd_cell
        if not component_plans:
            skipped_room_count += 1
            continue
        accepted_geometry = _mask_geometry(accepted_fill_mask)
        if accepted_geometry is None:
            skipped_room_count += 1
            continue
        room_area_m2 = float(room_mask.sum()) * resolution ** 2
        total_straight_lane_m = sum(
            _dist(start, finish)
            for _, _, _, cell_swaths, _ in component_plans
            for start, finish in cell_swaths
        )
        if (room_area_m2 < min_useful_region_area_m2
                or total_straight_lane_m < min_useful_region_lane_m):
            # A room-like fragment may be physically reachable yet provide
            # less straight cleaning than its entry/turn/perimeter overhead.
            # Keep it traversable for neighbouring rooms, but explicitly drop
            # it from the green target and route instead of drawing a knot.
            discarded_small_component_count += 1
            discarded_small_turn_safe |= accepted_fill_mask
            skipped_room_count += 1
            continue
        _, (r0, r1, c0, c1), _, _ = accepted_geometry
        component_plans.sort(
            key=lambda item: int(item[0].sum()), reverse=True)
        axis = component_plans[0][1]
        room_rows, room_cols = np.nonzero(room_mask)
        region_id = len(regions)
        centroid = (
            origin_x + (float(room_cols.mean()) + 0.5) * resolution,
            origin_y + (float(room_rows.mean()) + 0.5) * resolution,
        )
        regions.append(CoverageRegion(
            region_id=region_id,
            mask=accepted_fill_mask,
            bbox_cells=(r0, r1, c0, c1),
            centroid=centroid,
            area_m2=room_area_m2,
            axis=axis,
            cell_count=len(component_plans),
            travel_mask=room_mask,
            fragmentation_profiles=fragmentation_profiles,
        ))
        cells_by_region[region_id] = component_plans
    if not regions:
        return empty("no_region_has_cleaning_lanes", travel_reachable, snapped_start)

    raw_free = (grid >= 0) & (grid <= 50)
    for blocked in blocked_polygons or ():
        raw_free &= ~polygon_to_mask(
            blocked, width, height, resolution, origin_x, origin_y)

    global_path: List[Point] = []
    # All connectors in one plan share map resolution, width and body shape;
    # build the 16-heading footprint sweep tables once instead of per door.
    se2_stencil_table_cache: Dict[Tuple[object, ...], object] = {}
    segments: List[CoverageSegment] = []
    all_swaths: List[Swath] = []
    visit_order: List[int] = []
    failures: List[str] = []
    perimeter_issues: Dict[int, Tuple[int, int, int]] = {}
    component_issues: Dict[int, List[str]] = {}
    current = (
        robot_world if _point_is_free(
            travel_reachable, robot_world, resolution, origin_x, origin_y)
        else snapped_start
    )
    current_yaw = robot_yaw
    current_region_id = next((
        region.region_id for region in regions
        if region.travel_mask is not None
        and _point_is_free(
            region.travel_mask, current,
            resolution, origin_x, origin_y)
    ), -1)
    # Make one global sweep along the facility's long dimension.  From the
    # start, visit the nearer end first, reverse only once, then continue to the
    # far end.  The region containing the robot is not forced to be first: in a
    # central corridor that would clean the corridor, then traverse it again
    # for every side room.  Sorting every room by its station instead yields
    # upper rooms -> corridor -> lower rooms (or the reverse), so the long
    # corridor is covered once as part of the same monotonic sweep.
    span_x = max(region.centroid[0] for region in regions) - min(
        region.centroid[0] for region in regions)
    span_y = max(region.centroid[1] for region in regions) - min(
        region.centroid[1] for region in regions)
    progress_axis = 1 if span_y >= span_x else 0
    # A corridor with rooms on both sides can have a smaller centroid span than
    # the building's total left-right width.  Use the long axis of a degree>=3
    # adjacency hub so room stations are ordered along the corridor, not by
    # which side of it they occupy.
    region_degrees = {region.region_id: 0 for region in regions}
    region_adjacency: Dict[int, List[int]] = {
        region.region_id: [] for region in regions
    }
    expanded_region_masks = {
        region.region_id: dilate_binary(
            region.travel_mask, 1)
        for region in regions if region.travel_mask is not None
    }
    for first_index, first_region in enumerate(regions):
        if first_region.region_id not in expanded_region_masks:
            continue
        for second_region in regions[first_index + 1:]:
            if second_region.travel_mask is None:
                continue
            if not (expanded_region_masks[first_region.region_id]
                    & second_region.travel_mask).any():
                continue
            region_degrees[first_region.region_id] += 1
            region_degrees[second_region.region_id] += 1
            region_adjacency[first_region.region_id].append(
                second_region.region_id)
            region_adjacency[second_region.region_id].append(
                first_region.region_id)
    exit_aware_candidate_regions = set()
    for region in regions:
        neighbours = region_adjacency[region.region_id]
        geometry = (
            _mask_geometry(region.travel_mask)
            if region.travel_mask is not None else None)
        if (len(neighbours) == 1
                and region_degrees[neighbours[0]] >= 3
                and geometry is not None and geometry[2] >= 0.82):
            exit_aware_candidate_regions.add(region.region_id)
    exit_aware_leaf_regions = (
        exit_aware_candidate_regions if exit_aware_enabled else set())
    hub_region = max(
        regions,
        key=lambda region: (
            region_degrees[region.region_id],
            region.area_m2),
    )
    if region_degrees[hub_region.region_id] >= 3:
        hr0, hr1, hc0, hc1 = hub_region.bbox_cells
        progress_axis = 1 if (hr1 - hr0) >= (hc1 - hc0) else 0
    lateral_axis = 1 - progress_axis
    anchor_coord = current[progress_axis]
    lower = sorted(
        [region for region in regions
         if region.centroid[progress_axis] <= anchor_coord],
        key=lambda region: (
            -region.centroid[progress_axis], region.centroid[lateral_axis]),
    )
    upper = sorted(
        [region for region in regions
         if region.centroid[progress_axis] > anchor_coord],
        key=lambda region: (
            region.centroid[progress_axis], region.centroid[lateral_axis]),
    )
    lower_extent = (
        anchor_coord - min(region.centroid[progress_axis] for region in lower)
        if lower else float("inf"))
    upper_extent = (
        max(region.centroid[progress_axis] for region in upper) - anchor_coord
        if upper else float("inf"))
    if lower and (not upper or lower_extent <= upper_extent):
        station_region_sequence = [
            region.region_id for region in lower + upper]
    else:
        station_region_sequence = [
            region.region_id for region in upper + lower]

    def graph_chain(start_id: int, goal_id: int) -> List[int]:
        """Shortest region-adjacency chain used only for visit ordering."""
        if start_id == goal_id:
            return [start_id]
        if start_id not in region_adjacency or goal_id not in region_adjacency:
            return []
        parents: Dict[int, Optional[int]] = {start_id: None}
        queue = deque([start_id])
        while queue:
            region_id = queue.popleft()
            if region_id == goal_id:
                chain: List[int] = []
                cursor: Optional[int] = region_id
                while cursor is not None:
                    chain.append(cursor)
                    cursor = parents[cursor]
                return list(reversed(chain))
            for neighbour in region_adjacency[region_id]:
                if neighbour in parents:
                    continue
                parents[neighbour] = region_id
                queue.append(neighbour)
        return []

    # Candidate order obeys two operator-visible rules that a pure centroid
    # sort can violate: start with the region containing the robot, and never
    # drive through an uncleaned intermediate room/corridor to clean a farther
    # leaf first.  This produces long, monotonic first-entry sweeps on chains
    # while retaining the station order for branches that are not adjacent.
    current_first_sequence = list(station_region_sequence)
    if current_region_id in current_first_sequence:
        current_first_sequence.remove(current_region_id)
        current_first_sequence.insert(0, current_region_id)

    def insert_unvisited_graph_chain(
            seed_sequence: Sequence[int]) -> List[int]:
        ordered: List[int] = []
        seen = set()
        for target_id in seed_sequence:
            if target_id in seen:
                continue
            route_start = ordered[-1] if ordered else current_region_id
            chain = graph_chain(route_start, target_id)
            for region_id in chain[1:]:
                if region_id not in seen:
                    ordered.append(region_id)
                    seen.add(region_id)
            if target_id not in seen:
                ordered.append(target_id)
                seen.add(target_id)
        return ordered

    station_graph_sequence = insert_unvisited_graph_chain(
        station_region_sequence)
    current_graph_sequence = insert_unvisited_graph_chain(
        current_first_sequence)
    graph_order_recommended = any(
        candidate != station_region_sequence
        for candidate in (
            current_first_sequence,
            station_graph_sequence,
            current_graph_sequence))
    current_first_recommended = (
        current_first_sequence != station_region_sequence)
    order_sequences = {
        "station": station_region_sequence,
        "current_first": current_first_sequence,
        "station_graph": station_graph_sequence,
        "current_graph": current_graph_sequence,
    }
    if region_order_mode not in order_sequences:
        raise ValueError(f"unsupported region_order_mode={region_order_mode}")
    region_sequence = order_sequences[region_order_mode]

    def append_segment(segment: CoverageSegment) -> None:
        if len(segment.path) < 1:
            return
        if global_path and segment.path:
            join_gap = _dist(global_path[-1], segment.path[0])
            if (join_gap > 1e-7
                    and join_gap <= max(0.03, resolution * 0.75)):
                # Grid/world snapping can put two logically shared endpoints
                # a fraction of one cell apart.  The global polyline already
                # traverses that safe short edge; include it explicitly in the
                # next coloured segment so metadata/rendering has no visible
                # break and the continuity gate remains exact.
                segment.path = [global_path[-1]] + list(segment.path)
        if (global_path and segment.path
                and _dist(global_path[-1], segment.path[0]) <= 1e-7):
            segment.path_start_idx = len(global_path) - 1
        else:
            segment.path_start_idx = len(global_path)
        segments.append(segment)
        _extend_dedup(global_path, segment.path)
        segment.path_end_idx = len(global_path) - 1

    region_lookup = {region.region_id: region for region in regions}

    shortest_region_chain = graph_chain

    for region_sequence_index, chosen_id in enumerate(region_sequence):
        chosen = region_lookup[chosen_id]
        completed_route_penalty = np.zeros_like(
            travel_reachable, dtype=bool)
        if avoid_completed_route_transfers and global_path:
            completed_route_penalty = _polyline_cleaning_mask(
                travel_reachable.shape, global_path,
                resolution, origin_x, origin_y,
                max(resolution, min(0.18, swath_spacing_m * 0.35)))
        future_region_penalty = np.zeros_like(
            travel_reachable, dtype=bool)
        if avoid_future_region_swaths:
            for future_region_id in region_sequence[region_sequence_index:]:
                for (_, _, _, future_swaths,
                     _) in cells_by_region[future_region_id]:
                    for future_swath in future_swaths:
                        future_region_penalty |= _polyline_cleaning_mask(
                            travel_reachable.shape,
                            future_swath,
                            resolution,
                            origin_x,
                            origin_y,
                            max(
                                resolution,
                                min(0.18, swath_spacing_m * 0.35)),
                        )

        def transfer_penalty(start_point: Point,
                             goal_point: Point) -> Optional[np.ndarray]:
            if (not completed_route_penalty.any()
                    and not future_region_penalty.any()):
                return None
            penalty = completed_route_penalty | future_region_penalty
            clear_cells = max(
                1, int(math.ceil(0.45 / resolution)))
            for endpoint in (start_point, goal_point):
                row, col = _world_to_cell(
                    endpoint, resolution, origin_x, origin_y)
                penalty[
                    max(0, row - clear_cells):
                    min(height, row + clear_cells + 1),
                    max(0, col - clear_cells):
                    min(width, col + clear_cells + 1),
                ] = False
            return penalty

        def connect_transfer_orthogonal(
                scope: np.ndarray,
                start_point: Point,
                goal_point: Point) -> List[Point]:
            penalty = transfer_penalty(start_point, goal_point)
            if (hard_avoid_completed_route_transfers
                    and penalty is not None and penalty.any()):
                zero_crossing_scope = scope & ~penalty
                if (_point_is_free(
                        zero_crossing_scope, start_point,
                        resolution, origin_x, origin_y)
                        and _point_is_free(
                            zero_crossing_scope, goal_point,
                            resolution, origin_x, origin_y)):
                    zero_crossing = (
                        _connect_points_orthogonal_footprint_safe(
                            zero_crossing_scope,
                            raw_free,
                            start_point,
                            goal_point,
                            resolution,
                            origin_x,
                            origin_y,
                            path_step_m,
                            body,
                        ))
                    if zero_crossing:
                        return zero_crossing
            return _connect_points_orthogonal_footprint_safe(
                scope,
                raw_free,
                start_point,
                goal_point,
                resolution,
                origin_x,
                origin_y,
                path_step_m,
                body,
                penalty_mask=penalty,
            )
        # A single-door room must be left through the same doorway it entered.
        # Without an exit hint, lane ordering optimises only the first lane and
        # often finishes at the far wall; after the red perimeter loop the
        # transfer then cuts back across every completed swath.  Use only the
        # first portal on the already-fixed region graph route to the next
        # region.  This changes neither the region order nor the graph route;
        # it merely lets the last BCD cell choose which of its four valid
        # boustrophedon orientations should finish closer to that doorway.
        desired_perimeter_exit: Optional[Point] = None
        if region_sequence_index + 1 < len(region_sequence):
            next_region_id = region_sequence[region_sequence_index + 1]
            exit_chain = shortest_region_chain(chosen_id, next_region_id)
            if len(exit_chain) >= 2:
                next_hop_mask = region_lookup[exit_chain[1]].travel_mask
                if next_hop_mask is not None:
                    exit_rows, exit_cols = np.nonzero(
                        expanded_region_masks[chosen_id] & next_hop_mask)
                    if exit_cols.size:
                        desired_perimeter_exit = (
                            origin_x
                            + (float(exit_cols.mean()) + 0.5) * resolution,
                            origin_y
                            + (float(exit_rows.mean()) + 0.5) * resolution,
                        )
        desired_region_exit = (
            desired_perimeter_exit
            if chosen_id in exit_aware_leaf_regions else None)
        # Keep disconnected turn-safe islands inside one physical room grouped
        # rather than interleaving their lanes by a global row number.
        component_plans = list(cells_by_region[chosen.region_id])
        component_swath_count = sum(
            len(component[3]) for component in component_plans)
        monotonic_large_region = (
            adaptive_fragment_pruning
            and (
                (chosen.area_m2 >= 100.0
                 and len(component_plans) >= 4
                 and len(component_plans) < 20
                 and not (
                     len(regions) == 1 and len(component_plans) >= 12)
                 and len(component_plans)
                 <= max(1.0, component_swath_count * 0.30)
                 and chosen.axis in ("x", "y"))
                # A narrow, four-cell longitudinal room otherwise uses a
                # greedy next-cell choice that can switch columns midway.
                # Keep its BCD order monotonic so that one lateral transfer
                # is deferred to a room end instead of crossing the fill.
                or (40.0 <= chosen.area_m2 < 100.0
                    and len(component_plans) == 4
                    and chosen.axis == "y")))
        monotonic_perp_index = 1 if chosen.axis == "x" else 0
        monotonic_coordinates = [
            component[4][monotonic_perp_index]
            for component in component_plans
        ]
        monotonic_ascending = (
            not monotonic_coordinates
            or abs(current[monotonic_perp_index]
                   - min(monotonic_coordinates))
            <= abs(current[monotonic_perp_index]
                   - max(monotonic_coordinates)))
        component_masks = [component[0] for component in component_plans]
        component_se2_budget = (
            60_000 if len(component_masks) > 32 else 180_000)
        component_index_by_mask = {
            id(component_mask): index
            for index, component_mask in enumerate(component_masks)
        }
        component_adjacency: Dict[int, List[int]] = {
            index: [] for index in range(len(component_masks))
        }
        component_portals: Dict[
            Tuple[int, int], Tuple[np.ndarray, np.ndarray]
        ] = {}
        component_turn_safe_portals: Dict[
            Tuple[int, int], Tuple[np.ndarray, np.ndarray]
        ] = {}
        expanded_component_masks = [
            dilate_binary(component_mask, 1)
            for component_mask in component_masks
        ]
        component_bboxes: List[Tuple[int, int, int, int]] = []
        for component_mask in component_masks:
            component_rows, component_cols = np.nonzero(component_mask)
            component_bboxes.append((
                int(component_rows.min()), int(component_rows.max()),
                int(component_cols.min()), int(component_cols.max()),
            ))
        for first_index in range(len(component_masks)):
            for second_index in range(first_index + 1, len(component_masks)):
                fr0, fr1, fc0, fc1 = component_bboxes[first_index]
                sr0, sr1, sc0, sc1 = component_bboxes[second_index]
                if (fr1 + 1 < sr0 or sr1 + 1 < fr0
                        or fc1 + 1 < sc0 or sc1 + 1 < fc0):
                    continue
                portal_rows, portal_cols = np.nonzero(
                    expanded_component_masks[first_index]
                    & component_masks[second_index])
                if not portal_cols.size:
                    continue
                component_adjacency[first_index].append(second_index)
                component_adjacency[second_index].append(first_index)
                portal_x = origin_x + (portal_cols + 0.5) * resolution
                portal_y = origin_y + (portal_rows + 0.5) * resolution
                component_portals[(first_index, second_index)] = (
                    portal_x, portal_y)
                component_portals[(second_index, first_index)] = (
                    portal_x, portal_y)
                safe_portal = turn_safe[portal_rows, portal_cols]
                if safe_portal.any():
                    safe_xy = (
                        portal_x[safe_portal], portal_y[safe_portal])
                    component_turn_safe_portals[
                        (first_index, second_index)] = safe_xy
                    component_turn_safe_portals[
                        (second_index, first_index)] = safe_xy

        def component_chain(
            start_index: int,
            goal_index: int,
            allowed_indices: Sequence[int],
        ) -> List[int]:
            """Return an adjacency-only BCD transition through covered cells.

            A greedy cell order can finish a leaf before its sibling.  The old
            fallback then drew one long diagonal across the room.  Restricting
            the graph search to already covered cells plus the next cell makes
            that return explicit: leaf -> parent portal -> sibling portal.
            """
            allowed = set(int(index) for index in allowed_indices)
            if start_index not in allowed or goal_index not in allowed:
                return []
            parents: Dict[int, Optional[int]] = {start_index: None}
            queue = deque([start_index])
            while queue:
                current_index = queue.popleft()
                if current_index == goal_index:
                    chain: List[int] = []
                    cursor: Optional[int] = current_index
                    while cursor is not None:
                        chain.append(cursor)
                        cursor = parents[cursor]
                    return list(reversed(chain))
                for next_index in component_adjacency[current_index]:
                    if next_index not in allowed or next_index in parents:
                        continue
                    parents[next_index] = current_index
                    queue.append(next_index)
            return []

        ordered_components: List[
            Tuple[np.ndarray, str, float, List[Swath], Point, int]
        ] = []
        component_cursor = current
        previous_order_component_index: Optional[int] = None
        selected_component_indices = set()
        while component_plans:
            best_index = -1
            best_ordered: List[Swath] = []
            best_key = (
                2, float("inf"), float("inf"), float("inf"),
                float("inf"), float("inf"))
            for candidate_index, candidate in enumerate(component_plans):
                candidate_mask, candidate_axis, candidate_angle, raw_swaths, _ = candidate
                candidate_source_index = component_index_by_mask[
                    id(candidate_mask)]
                adjacent = bool(
                    previous_order_component_index is not None
                    and candidate_source_index in component_adjacency[
                        previous_order_component_index])
                graph_hops = 0
                frontier_penalty = 0
                if previous_order_component_index is not None and not adjacent:
                    frontier_nodes = {
                        node for node in selected_component_indices
                        if candidate_source_index in component_adjacency[node]
                    }
                    if not frontier_nodes:
                        frontier_penalty = 1
                        graph_hops = len(component_masks) + 1
                    else:
                        hop_queue = deque([
                            (previous_order_component_index, 0)])
                        hop_seen = {previous_order_component_index}
                        graph_hops = len(component_masks) + 1
                        while hop_queue:
                            node, hops = hop_queue.popleft()
                            if node in frontier_nodes:
                                graph_hops = hops + 1
                                break
                            for neighbour in component_adjacency[node]:
                                if (neighbour not in selected_component_indices
                                        or neighbour in hop_seen):
                                    continue
                                hop_seen.add(neighbour)
                                hop_queue.append((neighbour, hops + 1))
                final_component = len(component_plans) == 1
                forward_options = (
                    (True, False)
                    if final_component and desired_region_exit is not None
                    else (None,))
                for ascending in (True, False):
                    for initial_forward in forward_options:
                        candidate_ordered = _order_component_swaths(
                            raw_swaths, component_cursor,
                            candidate_axis, candidate_angle,
                            swath_spacing_m, ascending=ascending,
                            initial_forward=initial_forward)
                        if not candidate_ordered:
                            continue
                        entry_distance = _dist(
                            component_cursor, candidate_ordered[0][0])
                        exit_point = candidate_ordered[-1][1]
                        future_distances: List[float] = []
                        for future_index, future in enumerate(component_plans):
                            if future_index == candidate_index:
                                continue
                            future_source_index = component_index_by_mask[
                                id(future[0])]
                            portal_xy = component_portals.get(
                                (candidate_source_index, future_source_index))
                            if portal_xy is not None:
                                shared_x, shared_y = portal_xy
                                future_distances.append(float(np.sqrt(
                                    (shared_x - exit_point[0]) ** 2
                                    + (shared_y - exit_point[1]) ** 2).min()))
                            else:
                                future_distances.append(
                                    _dist(exit_point, future[4]) + 2.0)
                        future_distance = min(future_distances, default=0.0)
                        region_exit_distance = (
                            _dist(exit_point, desired_region_exit)
                            if final_component
                            and desired_region_exit is not None else 0.0)
                        monotonic_rank = 0.0
                        if (monotonic_large_region
                                and previous_order_component_index is not None):
                            coordinate = candidate[4][monotonic_perp_index]
                            monotonic_rank = (
                                coordinate if monotonic_ascending
                                else -coordinate)
                        key = (
                            frontier_penalty,
                            monotonic_rank,
                            graph_hops,
                            entry_distance + future_distance
                            + region_exit_distance,
                            entry_distance,
                            -int(candidate_mask.sum()),
                        )
                        if key < best_key:
                            best_key = key
                            best_index = candidate_index
                            best_ordered = candidate_ordered
            if best_index < 0 or not best_ordered:
                break
            component = component_plans.pop(best_index)
            ordered_components.append((
                component[0], component[1], component[2],
                best_ordered, component[4],
                component_index_by_mask[id(component[0])]))
            component_cursor = best_ordered[-1][1]
            previous_order_component_index = component_index_by_mask[
                id(component[0])]
            selected_component_indices.add(previous_order_component_index)
        if not ordered_components:
            failures.append(f"region_{chosen.region_id}_has_no_ordered_cells")
            continue
        first_mask, first_axis, first_angle, first_raw, _, _ = ordered_components[0]
        del first_mask
        del first_axis, first_angle
        first_ordered = first_raw
        if not first_ordered:
            failures.append(f"region_{chosen.region_id}_has_no_ordered_swaths")
            continue
        fill_start = first_ordered[0][0]
        transfer: List[Point] = []
        region_chain = shortest_region_chain(
            current_region_id, chosen.region_id)
        if len(region_chain) >= 2:
            routed: List[Point] = [current]
            route_cursor = current
            route_ok = True
            region_halo = max(
                1, int(math.ceil(0.25 / resolution)))
            for chain_offset, (from_id, to_id) in enumerate(
                    zip(region_chain, region_chain[1:])):
                from_mask = region_lookup[from_id].travel_mask
                to_mask = region_lookup[to_id].travel_mask
                if from_mask is None or to_mask is None:
                    route_ok = False
                    break
                portal_rows, portal_cols = np.nonzero(
                    expanded_region_masks[from_id] & to_mask)
                if not portal_cols.size:
                    route_ok = False
                    break
                portal_x = origin_x + (portal_cols + 0.5) * resolution
                portal_y = origin_y + (portal_rows + 0.5) * resolution
                if chain_offset + 2 < len(region_chain):
                    next_id = region_chain[chain_offset + 2]
                    next_rows, next_cols = np.nonzero(
                        expanded_region_masks[to_id]
                        & region_lookup[next_id].travel_mask)
                    if next_cols.size:
                        next_x = origin_x + (next_cols + 0.5) * resolution
                        next_y = origin_y + (next_rows + 0.5) * resolution
                        future_distance = np.sqrt(
                            (portal_x[:, None] - next_x[None, :]) ** 2
                            + (portal_y[:, None] - next_y[None, :]) ** 2,
                        ).min(axis=1)
                    else:
                        future_distance = np.hypot(
                            portal_x - fill_start[0],
                            portal_y - fill_start[1])
                else:
                    future_distance = np.hypot(
                        portal_x - fill_start[0],
                        portal_y - fill_start[1])
                portal_score = np.hypot(
                    portal_x - route_cursor[0],
                    portal_y - route_cursor[1]) + future_distance
                portal_index = int(np.argmin(portal_score))
                portal = (
                    float(portal_x[portal_index]),
                    float(portal_y[portal_index]))
                leg_scope = (
                    dilate_binary(from_mask, region_halo)
                    & travel_reachable)
                # Prefer the already completed room boundary when leaving a
                # high-rectangularity, single-door leaf.  This avoids drawing
                # a new chord across its finished parallel swaths.  The exact
                # body validator and the original full-room scope remain the
                # fallback if the boundary band is not connected.
                if (chain_offset == 0 and from_id == current_region_id
                        and from_id in exit_aware_leaf_regions):
                    exit_band_cells = max(
                        2, int(math.ceil(
                            max(0.45, swath_spacing_m) / resolution)))
                    exit_band = from_mask & ~erode_binary(
                        from_mask, exit_band_cells)
                    exit_scope = dilate_binary(exit_band, 1)
                    exit_scope |= (
                        expanded_region_masks[from_id] & to_mask)
                    exit_scope &= travel_reachable
                    if (_point_is_free(
                            exit_scope, route_cursor,
                            resolution, origin_x, origin_y)
                            and _point_is_free(
                                exit_scope, portal,
                                resolution, origin_x, origin_y)):
                        leg_scope = exit_scope
                leg = connect_transfer_orthogonal(
                    leg_scope, route_cursor, portal)
                if not leg:
                    leg = _connect_points_footprint_safe(
                        leg_scope, turn_safe & leg_scope, raw_free,
                        route_cursor, portal,
                        resolution, origin_x, origin_y, path_step_m,
                        body, rotation_safe_mask=rotation_safe_global,
                        stencil_table_cache=se2_stencil_table_cache)
                if not leg:
                    route_ok = False
                    break
                _extend_dedup(routed, leg[1:])
                route_cursor = portal
            if route_ok:
                destination_mask = region_lookup[chosen.region_id].travel_mask
                destination_scope = (
                    dilate_binary(destination_mask, region_halo)
                    & travel_reachable
                    if destination_mask is not None else travel_reachable)
                final_leg = connect_transfer_orthogonal(
                    destination_scope, route_cursor, fill_start)
                if not final_leg:
                    final_leg = _connect_points_footprint_safe(
                        destination_scope,
                        turn_safe & destination_scope,
                        raw_free,
                        route_cursor, fill_start,
                        resolution, origin_x, origin_y, path_step_m,
                        body, rotation_safe_mask=rotation_safe_global,
                        stencil_table_cache=se2_stencil_table_cache)
                if final_leg:
                    _extend_dedup(routed, final_leg[1:])
                    # Portal legs are individually continuous-safe.  Their
                    # shared door/corridor point may still require one explicit
                    # stop/rotate; validate that full-body rotation here and let
                    # the final hard-stop pass preserve it.  Rejecting the chain
                    # merely because it was not curvature-continuous launched a
                    # redundant facility-wide SE(2) search and could replace a
                    # door route with a cross-room shortcut.
                    route_valid, _ = _validate_connector_candidate_polyline(
                        raw_free, routed,
                        resolution, origin_x, origin_y,
                        footprint=body)
                    if route_valid:
                        transfer = routed
        transfer_scope = (
            global_travel_reachable
            if current_region_id < 0 else travel_reachable)
        if not transfer:
            transfer = connect_transfer_orthogonal(
                transfer_scope, current, fill_start)
        if not transfer:
            transfer = _connect_points_footprint_safe(
                transfer_scope,
                rotation_safe_global & transfer_scope,
                raw_free,
                current, fill_start,
                resolution, origin_x, origin_y, path_step_m,
                body, rotation_safe_mask=rotation_safe_global,
                stencil_table_cache=se2_stencil_table_cache,
            )
        if not transfer:
            failures.append(f"region_{chosen.region_id}_unreachable")
            continue
        if len(transfer) >= 2 and _dist(transfer[0], transfer[-1]) > 1e-6:
            append_segment(CoverageSegment(
                kind="transfer", region_id=chosen.region_id,
                path=transfer,
                from_region_id=current_region_id,
                to_region_id=chosen.region_id,
            ))
        local_travel = chosen.travel_mask if chosen.travel_mask is not None else travel_reachable
        fill_path: List[Point] = []
        ordered: List[Swath] = []
        component_swath_ranges: List[Tuple[int, int, int]] = []
        component_transition_diagnostics: List[Dict[str, object]] = []
        previous_fill_mask: Optional[np.ndarray] = None
        previous_component_index: Optional[int] = None
        covered_component_indices = set()
        for component_index, (component_mask, component_axis, component_angle,
                              component_raw, _, source_component_index) in enumerate(
                                  ordered_components):
            del component_axis, component_angle
            component_ordered = component_raw
            if not component_ordered:
                continue
            if fill_path:
                component_connector: List[Point] = []
                connector_strategy = "unresolved"
                connector_start_path_index = len(fill_path) - 1
                connector_from = fill_path[-1]
                connector_scope = local_travel
                connector_penalty = _polyline_cleaning_mask(
                    local_travel.shape, fill_path,
                    resolution, origin_x, origin_y,
                    max(resolution, min(0.18, swath_spacing_m * 0.35)))
                if avoid_future_component_swaths:
                    # Inter-cell returns are the main source of a connector
                    # cutting across lanes that have not yet been driven.  Add
                    # those future centre lines to the soft routing cost, so a
                    # safe headland route is preferred without ever making a
                    # physically reachable cell fail planning.
                    for future_component in ordered_components[
                            component_index:]:
                        for future_start, future_end in future_component[3]:
                            connector_penalty |= _polyline_cleaning_mask(
                                local_travel.shape,
                                [future_start, future_end],
                                resolution, origin_x, origin_y,
                                max(resolution, min(
                                    0.18, swath_spacing_m * 0.35)))
                traversed_penalty = connector_penalty.copy()
                penalty_clear_cells = max(
                    1, int(math.ceil(0.10 / resolution)))
                for endpoint in (
                        fill_path[-1], component_ordered[0][0]):
                    endpoint_row, endpoint_col = _world_to_cell(
                        endpoint, resolution, origin_x, origin_y)
                    connector_penalty[
                        max(0, endpoint_row - penalty_clear_cells):
                        min(height, endpoint_row + penalty_clear_cells + 1),
                        max(0, endpoint_col - penalty_clear_cells):
                        min(width, endpoint_col + penalty_clear_cells + 1),
                    ] = False

                def connector_overlap_cells(
                        candidate: Sequence[Point]) -> int:
                    sampled = _densify_polyline(
                        candidate, max(0.02, resolution * 0.50))
                    cells = {
                        _world_to_cell(
                            point, resolution, origin_x, origin_y)
                        for point in sampled
                    }
                    return sum(
                        0 <= row < height and 0 <= col < width
                        and bool(traversed_penalty[row, col])
                        for row, col in cells)

                if (previous_fill_mask is not None
                        and previous_component_index is not None):
                    local_halo = max(
                        1, int(math.ceil(0.45 / resolution)))
                    chain = component_chain(
                        previous_component_index,
                        source_component_index,
                        list(covered_component_indices)
                        + [source_component_index],
                    )
                    # First try a strict zero-crossing connector.  Completed
                    # swath centre lines are removed from the centre mask, but
                    # the headland around the union of already covered BCD
                    # cells remains available so the robot can go around lane
                    # endpoints instead of cutting across them.  Future cells
                    # are deliberately excluded: borrowing an uncleaned cell
                    # here would merely move the crossing to a later swath.
                    future_core = np.zeros_like(local_travel, dtype=bool)
                    future_core_inset = max(
                        2, int(math.ceil(
                            max(0.65, swath_spacing_m) / resolution)))
                    for future_index, future_mask in enumerate(component_masks):
                        if (future_index in covered_component_indices
                                or future_index == source_component_index):
                            continue
                        future_core |= erode_binary(
                            future_mask, future_core_inset)
                    no_cross_scope = (
                        local_travel
                        & ~connector_penalty
                        & ~future_core
                    )
                    no_cross_connector = []
                    accepted_no_cross: List[Point] = []
                    if (_point_is_free(
                            no_cross_scope, fill_path[-1],
                            resolution, origin_x, origin_y)
                            and _point_is_free(
                                no_cross_scope, component_ordered[0][0],
                                resolution, origin_x, origin_y)):
                        no_cross_connector = (
                            _connect_points_orthogonal_footprint_safe(
                                no_cross_scope, raw_free,
                                fill_path[-1], component_ordered[0][0],
                                resolution, origin_x, origin_y, path_step_m,
                                body))
                    if no_cross_connector:
                        no_cross_length = _partitioned_path_length(
                            no_cross_connector)
                        direct_distance = _dist(
                            fill_path[-1], component_ordered[0][0])
                        if no_cross_length <= max(
                                6.0, direct_distance * 2.0 + 6.0):
                            accepted_no_cross = no_cross_connector
                    graph_connector: List[Point] = []
                    if len(chain) >= 2:
                        routed: List[Point] = [fill_path[-1]]
                        route_cursor = fill_path[-1]
                        route_ok = True
                        edge_band_cells = max(
                            2, int(math.ceil(
                                max(0.45, swath_spacing_m) / resolution)))
                        for chain_offset, (from_index, to_index) in enumerate(
                                zip(chain, chain[1:])):
                            from_mask = component_masks[from_index]
                            to_mask = component_masks[to_index]
                            portal_xy = component_turn_safe_portals.get(
                                (from_index, to_index))
                            if portal_xy is None:
                                route_ok = False
                                break
                            portal_x, portal_y = portal_xy
                            if chain_offset + 2 < len(chain):
                                lookahead_index = chain[chain_offset + 2]
                                lookahead_xy = (
                                    component_turn_safe_portals.get(
                                        (to_index, lookahead_index)))
                                if lookahead_xy is not None:
                                    lookahead_x, lookahead_y = lookahead_xy
                                    # Portal boundaries are short; evaluating
                                    # all pairs remains cheap and avoids choosing
                                    # the wrong end of a doorway before a turn.
                                    next_distance = np.sqrt(
                                        (portal_x[:, None]
                                         - lookahead_x[None, :]) ** 2
                                        + (portal_y[:, None]
                                           - lookahead_y[None, :]) ** 2,
                                    ).min(axis=1)
                                else:
                                    next_distance = np.hypot(
                                        portal_x - component_ordered[0][0][0],
                                        portal_y - component_ordered[0][0][1])
                            else:
                                next_distance = np.hypot(
                                    portal_x - component_ordered[0][0][0],
                                    portal_y - component_ordered[0][0][1])
                            portal_score = np.hypot(
                                portal_x - route_cursor[0],
                                portal_y - route_cursor[1]) + next_distance
                            portal_index = int(np.argmin(portal_score))
                            portal = (
                                float(portal_x[portal_index]),
                                float(portal_y[portal_index]))

                            edge_band = from_mask & ~erode_binary(
                                from_mask, edge_band_cells)
                            edge_scope = dilate_binary(edge_band, 1)
                            edge_scope |= (
                                expanded_component_masks[from_index] & to_mask)
                            edge_scope &= local_travel
                            leg = _connect_points_orthogonal_footprint_safe(
                                edge_scope, raw_free, route_cursor, portal,
                                resolution, origin_x, origin_y, path_step_m,
                                body, penalty_mask=connector_penalty)
                            if not leg:
                                # Rasterised corners can split a thin boundary
                                # band.  Fall back only inside this one BCD cell,
                                # never to a facility-wide diagonal.
                                cell_scope = (
                                    dilate_binary(from_mask, local_halo)
                                    & local_travel)
                                leg = _connect_points_orthogonal_footprint_safe(
                                    cell_scope, raw_free,
                                    route_cursor, portal,
                                    resolution, origin_x, origin_y, path_step_m,
                                    body, penalty_mask=connector_penalty)
                            if not leg:
                                route_ok = False
                                break
                            _extend_dedup(routed, leg[1:])
                            route_cursor = portal
                        if route_ok:
                            destination_band = component_mask & ~erode_binary(
                                component_mask, edge_band_cells)
                            destination_scope = dilate_binary(
                                destination_band, 1) & local_travel
                            final_leg = _connect_points_orthogonal_footprint_safe(
                                destination_scope, raw_free, route_cursor,
                                component_ordered[0][0],
                                resolution, origin_x, origin_y, path_step_m,
                                body, penalty_mask=connector_penalty)
                            if not final_leg:
                                destination_scope = (
                                    dilate_binary(component_mask, local_halo)
                                    & local_travel)
                                final_leg = _connect_points_orthogonal_footprint_safe(
                                    destination_scope, raw_free, route_cursor,
                                    component_ordered[0][0],
                                    resolution, origin_x, origin_y,
                                    path_step_m, body,
                                    penalty_mask=connector_penalty)
                            if final_leg:
                                _extend_dedup(routed, final_leg[1:])
                                graph_connector = routed
                    if graph_connector:
                        component_connector = graph_connector
                        connector_strategy = "component_graph"
                    if accepted_no_cross:
                        use_no_cross = not component_connector
                        if component_connector:
                            no_cross_overlap = connector_overlap_cells(
                                accepted_no_cross)
                            graph_overlap = connector_overlap_cells(
                                component_connector)
                            no_cross_length = _partitioned_path_length(
                                accepted_no_cross)
                            graph_length = _partitioned_path_length(
                                component_connector)
                            use_no_cross = (
                                no_cross_overlap < graph_overlap
                                and no_cross_length
                                <= max(6.0, graph_length * 1.25 + 1.0)
                            )
                        if use_no_cross:
                            component_connector = accepted_no_cross
                            connector_strategy = "zero_crossing"
                    local_scope = dilate_binary(
                        previous_fill_mask | component_mask, local_halo)
                    local_scope &= local_travel
                    if (_point_is_free(
                            local_scope, fill_path[-1],
                            resolution, origin_x, origin_y)
                            and _point_is_free(
                                local_scope, component_ordered[0][0],
                                resolution, origin_x, origin_y)):
                        connector_scope = local_scope
                if not component_connector:
                    component_connector = _connect_points_orthogonal_footprint_safe(
                        connector_scope, raw_free,
                        fill_path[-1], component_ordered[0][0],
                        resolution, origin_x, origin_y, path_step_m,
                        body, penalty_mask=connector_penalty)
                    if component_connector:
                        connector_strategy = "local_orthogonal"
                if not component_connector:
                    # The narrow local halo is useful for cheap orthogonal
                    # routing, but an SE(2) search in it repeatedly exhausts
                    # its budget before the exact same search succeeds in the
                    # physical-room mask.  Go directly to that one bounded
                    # forward-only search.
                    component_connector = _connect_points_footprint_safe(
                        local_travel, turn_safe, raw_free,
                        fill_path[-1], component_ordered[0][0],
                        resolution, origin_x, origin_y, path_step_m,
                        body, max_se2_expansions=component_se2_budget,
                        rotation_safe_mask=rotation_safe_global,
                        stencil_table_cache=se2_stencil_table_cache,
                        penalty_mask=connector_penalty)
                    if component_connector:
                        connector_strategy = "room_se2"
                if not component_connector:
                    component_issues.setdefault(
                        chosen.region_id, []).append(
                            f"{component_index}:unreachable")
                    continue
                connector_overlap = connector_overlap_cells(
                    component_connector)
                _extend_dedup(fill_path, component_connector[1:])
                component_transition_diagnostics.append({
                    "from_component_id": int(previous_component_index),
                    "to_component_id": int(source_component_index),
                    "from_swath_index": int(len(ordered) - 1),
                    "to_swath_index": int(len(ordered)),
                    "path_start_index": int(connector_start_path_index),
                    "path_end_index": int(len(fill_path) - 1),
                    "strategy": connector_strategy,
                    "length_m": float(_partitioned_path_length(
                        component_connector)),
                    "overlap_cells": int(connector_overlap),
                    "start": (float(connector_from[0]),
                              float(connector_from[1])),
                    "end": (float(component_ordered[0][0][0]),
                            float(component_ordered[0][0][1])),
                })
            component_swath_start = len(ordered)
            component_path = _stitch_path(
                free_mask=component_mask,
                entry_free_mask=component_mask,
                swaths=component_ordered,
                start=component_ordered[0][0],
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
                path_step_m=path_step_m,
                penalty_mask=None,
                semicircle_uturns=False,
                avoid_traversed_lanes=True,
            )
            if not component_path:
                component_issues.setdefault(
                    chosen.region_id, []).append(
                        f"{component_index}:stitch_failed")
                continue
            _extend_dedup(fill_path, component_path)
            ordered.extend(component_ordered)
            component_swath_ranges.append((
                int(source_component_index),
                int(component_swath_start),
                int(len(ordered)),
            ))
            previous_fill_mask = component_mask
            previous_component_index = source_component_index
            covered_component_indices.add(source_component_index)
        if not fill_path:
            failures.append(f"region_{chosen.region_id}_fill_stitch_failed")
            current = transfer[-1]
            continue

        # Commit the full blue fill before beginning the red edge-clean phase.
        # A physical room may contain several disconnected turn-safe islands
        # around furniture.  Trace one outer ring per cleaned island instead of
        # keeping only the globally largest ring (the source of "red only in a
        # small room").  Blue body-safe connectors join those red rings.
        append_segment(CoverageSegment(
            kind="fill", region_id=chosen.region_id,
            path=fill_path,
            from_region_id=chosen.region_id,
            to_region_id=chosen.region_id,
            swaths=list(ordered),
            component_swath_ranges=list(component_swath_ranges),
            component_transition_diagnostics=list(
                component_transition_diagnostics),
        ))
        # Rectangular cells get one close, tangent-continuous loop.  It is
        # accepted only after the same 0.55 m lookahead body sweep used by the
        # controller path passes with zero violations.  Irregular/cluttered
        # cells retain the conservative turn-safe contour fallback.
        rounded_ring = _plan_near_rectangular_perimeter(
            local_travel,
            raw_free,
            resolution,
            origin_x,
            origin_y,
            path_step_m,
            body,
            clean_width_m,
        )
        mandatory_rings: List[List[Point]] = (
            [rounded_ring] if rounded_ring else [])
        optional_rings: List[List[Point]] = []
        optional_ring_sources: Dict[
            int, Tuple[np.ndarray, List[Point]]
        ] = {}
        mandatory_ring_fallbacks: Dict[int, List[Point]] = {}
        # BCD cell boundaries are planning cuts, not physical walls.  Edge
        # cleaning traces the original turn-safe room boundary; obstacle-hole
        # rings and lidar-created islands are completion candidates only when
        # they add otherwise uncovered floor.  When a rounded outer room loop
        # is available it replaces the largest outer contour.
        perimeter_components = _connected_components_fast(
            chosen.mask,
            min_cells=max(
                1, int(math.ceil(0.20 / resolution ** 2))),
        )
        perimeter_components.sort(
            key=lambda component: int(component.sum()), reverse=True)
        rounded_fallback_pending = bool(rounded_ring)
        mandatory_outer_assigned = bool(rounded_ring)
        for component_mask in perimeter_components:
            component_rings = _trace_perimeter_rings(
                component_mask, resolution, origin_x, origin_y,
                step_m=path_step_m, min_ring_cells=8,
            )
            if not component_rings:
                continue
            ordered_rings = sorted(
                component_rings,
                key=lambda candidate: _polyline_area(
                    candidate, path_step_m * 2.5),
                reverse=True)
            rounded_fallback_index = (
                0 if rounded_fallback_pending and ordered_rings else -1)
            if rounded_fallback_index >= 0:
                rounded_fallback_pending = False
            for ring_index, ring in enumerate(ordered_rings):
                is_rounded_fallback = ring_index == rounded_fallback_index
                mandatory_candidate = (
                    ring_index == 0
                    and not is_rounded_fallback
                    and not mandatory_outer_assigned)
                source_ring = list(ring)
                elastic_ring = (
                    _elasticize_perimeter_ring(
                        source_ring,
                        component_mask,
                        local_travel,
                        travel_reachable,
                        turn_safe,
                        raw_free,
                        resolution,
                        origin_x,
                        origin_y,
                        path_step_m,
                        body,
                        clean_width_m,
                    )
                    if (mandatory_candidate or is_rounded_fallback) else [])
                if len(ring) >= 3:
                    ring = _rdp_masked(
                        ring, max(0.10, resolution * 1.5),
                        component_mask, resolution, origin_x, origin_y)
                    ring = _densify_polyline(ring, path_step_m)
                if elastic_ring:
                    ring = max(
                        (ring, elastic_ring),
                        key=lambda candidate: int((
                            _polyline_cleaning_mask(
                                local_travel.shape, candidate,
                                resolution, origin_x, origin_y,
                                clean_width_m)
                            & local_travel).sum()),
                    )
                if (ring and sum(_dist(first, second)
                                 for first, second in zip(
                                     ring, ring[1:])) >= 0.60):
                    # The largest ring of every original component is its room
                    # boundary.  Only the largest room boundary is mandatory;
                    # outer contours of smaller turn-safe islands and obstacle
                    # holes are optional completion passes.  Treating every
                    # noisy island as a wall was the source of tangled red
                    # loops on saved lidar maps.
                    if is_rounded_fallback and rounded_ring:
                        mandatory_ring_fallbacks[id(rounded_ring)] = ring
                    elif mandatory_candidate:
                        mandatory_rings.append(ring)
                        mandatory_outer_assigned = True
                    else:
                        optional_rings.append(ring)
                        optional_ring_sources[id(ring)] = (
                            component_mask, source_ring)

        local_target = (
            _disk_dilate_mask(
                chosen.mask, resolution, max(0.0, clean_width_m * 0.5))
            & local_travel & mission_mask
        )
        selected_cleaned = _polyline_cleaning_mask(
            local_travel.shape, fill_path,
            resolution, origin_x, origin_y, clean_width_m)
        for ring in mandatory_rings:
            selected_cleaned |= _polyline_cleaning_mask(
                local_travel.shape, ring,
                resolution, origin_x, origin_y, clean_width_m)
        selected_optional: List[List[Point]] = []
        target_cells = int(local_target.sum())
        while optional_rings and target_cells:
            local_ratio = float((selected_cleaned & local_target).sum()) / target_cells
            if local_ratio >= 0.952:
                break
            best_index = -1
            best_key = (-1.0, -1)
            best_mask: Optional[np.ndarray] = None
            for candidate_index, candidate in enumerate(optional_rings):
                candidate_mask = _polyline_cleaning_mask(
                    local_travel.shape, candidate,
                    resolution, origin_x, origin_y, clean_width_m)
                gain = int((
                    candidate_mask & local_target & ~selected_cleaned).sum())
                candidate_length = sum(
                    _dist(first, second)
                    for first, second in zip(candidate, candidate[1:]))
                key = (gain / max(0.50, candidate_length), gain)
                if key > best_key:
                    best_index = candidate_index
                    best_key = key
                    best_mask = candidate_mask
            if best_index < 0 or best_key[1] <= 0 or best_mask is None:
                break
            selected_ring = optional_rings.pop(best_index)
            source = optional_ring_sources.pop(id(selected_ring), None)
            if source is not None:
                source_component, source_ring = source
                elastic_ring = _elasticize_perimeter_ring(
                    source_ring,
                    source_component,
                    local_travel,
                    travel_reachable,
                    turn_safe,
                    raw_free,
                    resolution,
                    origin_x,
                    origin_y,
                    path_step_m,
                    body,
                    clean_width_m,
                )
                if elastic_ring:
                    selected_ring = max(
                        (selected_ring, elastic_ring),
                        key=lambda candidate: (
                            int((
                                _polyline_cleaning_mask(
                                    local_travel.shape, candidate,
                                    resolution, origin_x, origin_y,
                                    clean_width_m)
                                & local_target & ~selected_cleaned).sum()),
                            -sum(_dist(first, second)
                                 for first, second in zip(
                                     candidate, candidate[1:])),
                        ),
                    )
            best_mask = _polyline_cleaning_mask(
                local_travel.shape, selected_ring,
                resolution, origin_x, origin_y, clean_width_m)
            selected_optional.append(selected_ring)
            selected_cleaned |= best_mask
        # A turn-safe circular contour is intentionally conservative beside a
        # straight wall because TD25A has a long rear overhang.  If a region is
        # still below the coverage gate, add only the exact-body-safe physical
        # wall arcs that contribute new cells.  Artificial room/BCD seams are
        # excluded by ``_extract_wall_follow_arcs`` and every arc is validated
        # at its true tangent yaw before it can become a red segment.
        selected_wall_arcs: List[List[Point]] = []
        wall_arc_escapes: Dict[int, List[Point]] = {}
        local_ratio = (
            float((selected_cleaned & local_target).sum()) / target_cells
            if target_cells else 1.0)
        local_wall_geometry = _mask_geometry(local_travel)
        raw_wall_arcs = (
            _extract_wall_follow_arcs(
                local_travel,
                travel_reachable,
                raw_free,
                resolution,
                origin_x,
                origin_y,
                path_step_m,
                body,
            )
            if (local_ratio < 0.952
                and local_wall_geometry is not None
                and local_wall_geometry[2] >= 0.88) else [])
        wall_arcs: List[List[Point]] = []
        for raw_wall_arc in raw_wall_arcs:
            wall_arc, escape = _prepare_forward_wall_arc(
                raw_wall_arc,
                travel_reachable,
                turn_safe,
                raw_free,
                resolution,
                origin_x,
                origin_y,
                path_step_m,
                body,
            )
            if not wall_arc or not escape:
                continue
            wall_arcs.append(wall_arc)
            wall_arc_escapes[id(wall_arc)] = escape
        while wall_arcs and target_cells:
            local_ratio = float((selected_cleaned & local_target).sum()) / target_cells
            if local_ratio >= 0.952:
                break
            best_index = -1
            best_key = (-1.0, -1)
            best_mask = None
            for candidate_index, candidate in enumerate(wall_arcs):
                candidate_mask = _polyline_cleaning_mask(
                    local_travel.shape, candidate,
                    resolution, origin_x, origin_y, clean_width_m)
                gain = int((
                    candidate_mask & local_target & ~selected_cleaned).sum())
                candidate_length = sum(
                    _dist(first, second)
                    for first, second in zip(candidate, candidate[1:]))
                key = (gain / max(0.50, candidate_length), gain)
                if key > best_key:
                    best_index = candidate_index
                    best_key = key
                    best_mask = candidate_mask
            if best_index < 0 or best_key[1] <= 0 or best_mask is None:
                break
            selected_wall_arcs.append(wall_arcs.pop(best_index))
            selected_cleaned |= best_mask
        pending_rings: List[List[Point]] = (
            mandatory_rings + selected_optional + selected_wall_arcs)
        mandatory_ring_ids = {id(ring) for ring in mandatory_rings}

        perimeter_cursor = fill_path[-1]
        perimeter_yaw = _terminal_path_yaw(fill_path, current_yaw)
        fill_to_perimeter_penalty = (
            _polyline_cleaning_mask(
                local_travel.shape,
                fill_path,
                resolution,
                origin_x,
                origin_y,
                max(resolution, min(0.18, swath_spacing_m * 0.35)),
            )
            if avoid_fill_to_perimeter_crossings else None)

        def perimeter_entry_penalty(goal_point: Point) -> Optional[np.ndarray]:
            if fill_to_perimeter_penalty is None:
                return None
            penalty = fill_to_perimeter_penalty.copy()
            clear_cells = max(
                1, int(math.ceil(0.45 / resolution)))
            for endpoint in (perimeter_cursor, goal_point):
                row, col = _world_to_cell(
                    endpoint, resolution, origin_x, origin_y)
                penalty[
                    max(0, row - clear_cells):
                    min(height, row + clear_cells + 1),
                    max(0, col - clear_cells):
                    min(width, col + clear_cells + 1),
                ] = False
            return penalty

        completed_perimeters = 0
        while pending_rings:
            pending_rings.sort(
                key=lambda ring: min(
                    _dist(perimeter_cursor, point) for point in ring))
            connected_index = -1
            connected_ring: List[Point] = []
            connected_path: List[Point] = []
            connected_escape: List[Point] = []
            for candidate_index, raw_ring in enumerate(pending_rings):
                prepared_escape = wall_arc_escapes.get(id(raw_ring), [])
                if prepared_escape:
                    # A close wall arc is one-way for the asymmetric body; do
                    # not reverse it merely because the other endpoint is near.
                    candidate_variants = [list(raw_ring)]
                else:
                    candidate_variants = _ring_start_variants(
                        raw_ring,
                        perimeter_cursor,
                        path_step_m,
                        turn_safe,
                        resolution,
                        origin_x,
                        origin_y,
                    )
                    if (avoid_fill_to_perimeter_crossings
                            and desired_perimeter_exit is not None):
                        exit_variants = _ring_start_variants(
                            raw_ring,
                            desired_perimeter_exit,
                            path_step_m,
                            turn_safe,
                            resolution,
                            origin_x,
                            origin_y,
                        )
                        seen_starts = {
                            (round(variant[0][0], 6),
                             round(variant[0][1], 6))
                            for variant in candidate_variants if variant
                        }
                        for exit_variant in exit_variants:
                            if not exit_variant:
                                continue
                            key = (
                                round(exit_variant[0][0], 6),
                                round(exit_variant[0][1], 6),
                            )
                            if key in seen_starts:
                                continue
                            seen_starts.add(key)
                            candidate_variants.append(exit_variant)
                if not candidate_variants:
                    continue
                boundary_connector: List[Point] = []
                candidate: List[Point] = []
                if prepared_escape:
                    candidate_variant = candidate_variants[0]
                    trial_connector = _connect_points_footprint_safe(
                        travel_reachable, turn_safe, raw_free,
                        perimeter_cursor, candidate_variant[0],
                        resolution, origin_x, origin_y, path_step_m,
                        body,
                        start_yaw=perimeter_yaw,
                        goal_yaw=_initial_path_yaw(candidate_variant),
                        max_se2_expansions=60_000,
                        rotation_safe_mask=rotation_safe_global,
                        stencil_table_cache=se2_stencil_table_cache,
                    )
                    if trial_connector:
                        connector_valid, _ = validate_footprint_path(
                            raw_free, trial_connector,
                            resolution, origin_x, origin_y,
                            footprint=body, lookahead_m=0.0)
                        if connector_valid:
                            candidate = candidate_variant
                            boundary_connector = trial_connector
                else:
                    # Try every cheap axis-aligned start first.  Launching an
                    # SE(2) lattice search for each of eight ring rotations was
                    # the source of the 30--40 second map-9 regression.
                    best_variant_key = (
                        float("inf"), float("inf"), float("inf"),
                        float("inf"), float("inf"))
                    for candidate_variant in candidate_variants:
                        active_entry_penalty = perimeter_entry_penalty(
                            candidate_variant[0])
                        trial_connector = (
                            _connect_points_orthogonal_footprint_safe(
                                turn_safe, raw_free,
                                perimeter_cursor, candidate_variant[0],
                                resolution, origin_x, origin_y, path_step_m,
                                body,
                                penalty_mask=active_entry_penalty))
                        if not trial_connector:
                            continue
                        connector_valid, _ = validate_footprint_path(
                            raw_free, trial_connector,
                            resolution, origin_x, origin_y,
                            footprint=body, lookahead_m=0.0)
                        if connector_valid:
                            if active_entry_penalty is None:
                                candidate = candidate_variant
                                boundary_connector = trial_connector
                                break
                            connector_mask = _polyline_cleaning_mask(
                                local_travel.shape,
                                trial_connector,
                                resolution,
                                origin_x,
                                origin_y,
                                max(resolution, 0.12),
                            )
                            overlap = int((
                                connector_mask & active_entry_penalty).sum())
                            connector_length = _partitioned_path_length(
                                trial_connector)
                            proper_crossings = (
                                _proper_polyline_crossing_count(
                                    trial_connector, fill_path))
                            exit_distance = (
                                _dist(
                                    candidate_variant[0],
                                    desired_perimeter_exit)
                                if desired_perimeter_exit is not None else 0.0)
                            variant_key = (
                                proper_crossings,
                                overlap,
                                connector_length + exit_distance,
                                exit_distance,
                                connector_length,
                            )
                            if variant_key < best_variant_key:
                                best_variant_key = variant_key
                                candidate = candidate_variant
                                boundary_connector = trial_connector
                            if (overlap == 0 and proper_crossings == 0
                                    and desired_perimeter_exit is None):
                                break
                    if not boundary_connector:
                        for candidate_variant in candidate_variants[:1]:
                            trial_connector = _connect_points_footprint_safe(
                                local_travel, turn_safe, raw_free,
                                perimeter_cursor, candidate_variant[0],
                                resolution, origin_x, origin_y, path_step_m,
                                body, max_se2_expansions=60_000,
                                rotation_safe_mask=rotation_safe_global,
                                stencil_table_cache=se2_stencil_table_cache)
                            if not trial_connector:
                                continue
                            connector_valid, _ = validate_footprint_path(
                                raw_free, trial_connector,
                                resolution, origin_x, origin_y,
                                footprint=body, lookahead_m=0.0)
                            if connector_valid:
                                candidate = candidate_variant
                                boundary_connector = trial_connector
                                break
                if boundary_connector:
                    connected_index = candidate_index
                    connected_ring = candidate
                    connected_path = boundary_connector
                    connected_escape = prepared_escape
                    break
            if connected_index < 0:
                fallback_applied = False
                for fallback_index, failed_ring in enumerate(pending_rings):
                    fallback_ring = mandatory_ring_fallbacks.pop(
                        id(failed_ring), None)
                    if not fallback_ring:
                        continue
                    was_mandatory = id(failed_ring) in mandatory_ring_ids
                    pending_rings[fallback_index] = fallback_ring
                    if was_mandatory:
                        mandatory_ring_ids.remove(id(failed_ring))
                        mandatory_ring_ids.add(id(fallback_ring))
                    fallback_applied = True
                    break
                if fallback_applied:
                    # The close rounded loop itself was safe but no forward
                    # connector could reach its tangent start.  Retry once with
                    # the conservative turn-safe contour of the same physical
                    # outer wall, never with an artificial BCD boundary.
                    continue
                break
            pending_rings.pop(connected_index)
            if (len(connected_path) >= 2
                    and _dist(connected_path[0], connected_path[-1]) > 1e-6):
                append_segment(CoverageSegment(
                    kind="transfer", region_id=chosen.region_id,
                    path=connected_path,
                    from_region_id=chosen.region_id,
                    to_region_id=chosen.region_id,
                ))
            append_segment(CoverageSegment(
                kind="perimeter", region_id=chosen.region_id,
                path=connected_ring,
                from_region_id=chosen.region_id,
                to_region_id=chosen.region_id,
                continuous_to_next=bool(connected_escape),
            ))
            perimeter_cursor = connected_ring[-1]
            perimeter_yaw = _terminal_path_yaw(
                connected_ring, perimeter_yaw)
            if connected_escape:
                append_segment(CoverageSegment(
                    kind="transfer", region_id=chosen.region_id,
                    path=connected_escape,
                    from_region_id=chosen.region_id,
                    to_region_id=chosen.region_id,
                ))
                perimeter_cursor = connected_escape[-1]
                perimeter_yaw = _terminal_path_yaw(
                    connected_escape, perimeter_yaw)
            completed_perimeters += 1
        missing_mandatory = sum(
            id(ring) in mandatory_ring_ids for ring in pending_rings)
        optional_missing = len(pending_rings) - missing_mandatory
        if completed_perimeters == 0 or missing_mandatory or optional_missing:
            perimeter_issues[chosen.region_id] = (
                completed_perimeters, missing_mandatory, optional_missing)
        current = perimeter_cursor
        current_yaw = perimeter_yaw
        all_swaths.extend(ordered)
        visit_order.append(chosen.region_id)
        current_region_id = chosen.region_id

    # BCD and room masks exist only to order interior fill.  Their boundaries
    # are not walls, so their already-certified loops become transit paths and
    # one perimeter is traced from the global turn-safe component instead.
    for segment in segments:
        if segment.kind == "perimeter":
            segment.kind = "transfer"
    global_components = _connected_components_fast(
        turn_safe, min_cells=max(1, int(math.ceil(0.20 / resolution ** 2))))
    global_ring: List[Point] = []
    if global_components:
        global_component = max(global_components, key=lambda item: int(item.sum()))
        global_rings = _trace_perimeter_rings(
            global_component, resolution, origin_x, origin_y,
            step_m=path_step_m, min_ring_cells=8)
        if global_rings:
            source_ring = max(
                global_rings,
                key=lambda ring: _polyline_area(ring, path_step_m * 2.5))
            global_ring = _elasticize_perimeter_ring(
                source_ring, global_component, travel_reachable,
                travel_reachable, turn_safe, raw_free, resolution,
                origin_x, origin_y, path_step_m, body, clean_width_m)
            if not global_ring:
                global_ring = _densify_polyline(_rdp_masked(
                    source_ring, max(0.10, resolution * 1.5),
                    global_component, resolution, origin_x, origin_y),
                    path_step_m)
    if global_ring:
        variants = _ring_start_variants(
            global_ring, global_path[-1], path_step_m, turn_safe,
            resolution, origin_x, origin_y)
        for variant in variants:
            connector = _connect_points_footprint_safe(
                travel_reachable, turn_safe, raw_free, global_path[-1],
                variant[0], resolution, origin_x, origin_y, path_step_m,
                body, rotation_safe_mask=rotation_safe_global,
                stencil_table_cache=se2_stencil_table_cache)
            valid, _ = validate_footprint_path(
                raw_free, variant, resolution, origin_x, origin_y,
                footprint=body, lookahead_m=0.55)
            if connector and valid:
                if _dist(connector[0], connector[-1]) > 1e-6:
                    append_segment(CoverageSegment(
                        kind="transfer", region_id=visit_order[-1],
                        path=connector, from_region_id=visit_order[-1],
                        to_region_id=visit_order[-1]))
                append_segment(CoverageSegment(
                    kind="perimeter", region_id=visit_order[-1],
                    path=variant, from_region_id=visit_order[-1],
                    to_region_id=visit_order[-1]))
                break
    if not any(segment.kind == "perimeter" for segment in segments):
        failures.append("global_perimeter_missing")

    refined_transfer_count = 0
    refined_transfer_crossing_reduction = 0
    refinement_diagnostics: Dict[str, int] = {}

    def note_refinement(name: str) -> None:
        refinement_diagnostics[name] = (
            refinement_diagnostics.get(name, 0) + 1)

    # Component-to-component connectors live inside one yellow fill segment,
    # so the older transfer-only post-pass could not improve them.  Once every
    # room lane and perimeter is fixed, retry only connectors that actually
    # cross planned geometry.  Endpoints, swath order and region order remain
    # immutable; a replacement must be full-body safe and Pareto-better on
    # crossings without an excessive length or turn regression.
    component_connector_refined = False
    if (refine_inter_region_transfers
            and refine_same_region_transfers
            and len(segments) > 2):
        component_penalty_width_m = max(
            resolution, min(0.12, swath_spacing_m * 0.25))
        fixed_penalty_paths: List[Sequence[Point]] = []
        component_crossing_references: List[Sequence[Point]] = []
        for fixed_segment in segments:
            if fixed_segment.kind == "fill":
                for fixed_swath in fixed_segment.swaths:
                    swath_path = [fixed_swath[0], fixed_swath[1]]
                    fixed_penalty_paths.append(swath_path)
                    component_crossing_references.append(swath_path)
            elif (fixed_segment.kind == "perimeter"
                    and not refine_fill_only_penalty):
                fixed_penalty_paths.append(fixed_segment.path)
            elif (fixed_segment.kind == "transfer"
                    and len(fixed_segment.path) >= 2):
                component_crossing_references.append(
                    fixed_segment.path)
                if refine_against_all_routes:
                    fixed_penalty_paths.append(fixed_segment.path)
        fixed_cleaning_penalty = _polylines_cleaning_mask(
            travel_reachable.shape,
            fixed_penalty_paths,
            resolution,
            origin_x,
            origin_y,
            component_penalty_width_m,
        )
        component_endpoint_clear_cells = max(
            1, int(math.ceil(0.45 / resolution)))
        component_refine_halo_cells = max(
            1, int(math.ceil(0.25 / resolution)))
        component_refine_body_samples = _footprint_samples(
            body, resolution)
        for fill_segment_index, fill_segment in enumerate(segments):
            if (fill_segment.kind != "fill"
                    or not fill_segment.component_transition_diagnostics):
                continue
            region = region_lookup.get(fill_segment.region_id)
            if region is None or region.travel_mask is None:
                note_refinement("component_invalid_region_scope")
                continue
            component_route_scope = (
                dilate_binary(
                    region.travel_mask, component_refine_halo_cells)
                & travel_reachable)
            transitions = sorted(
                fill_segment.component_transition_diagnostics,
                key=lambda item: int(item["path_start_index"]),
                reverse=True,
            )
            # Score against immutable swaths/transfers before variable-length
            # connector slices are replaced.  Orthogonal retries remain cheap;
            # the costly SE(2) retry is budgeted to the strongest one or two
            # candidates in each physical region.
            se2_ranked_candidates: List[Tuple[int, int, int]] = []
            for transition in transitions:
                score_start = int(transition["path_start_index"])
                score_end = int(transition["path_end_index"])
                if (score_start < 0 or score_end <= score_start
                        or score_end >= len(fill_segment.path)):
                    continue
                score_connector = fill_segment.path[
                    score_start:score_end + 1]
                score = _proper_polyline_crossing_count_many(
                    score_connector, component_crossing_references)
                transition["refinement_crossings_before"] = int(score)
                if (score >= 6
                        and transition["strategy"] != "room_se2"
                        and int(transition["overlap_cells"]) <= 100):
                    se2_ranked_candidates.append((
                        int(score), score_start, id(transition)))
            se2_retry_budget = 2 if region.cell_count >= 20 else 1
            se2_retry_transition_ids = {
                transition_id
                for _, _, transition_id in sorted(
                    se2_ranked_candidates,
                    key=lambda item: (-item[0], item[1]),
                )[:se2_retry_budget]
            }
            for transition in transitions:
                connector_start = int(transition["path_start_index"])
                connector_end = int(transition["path_end_index"])
                if (connector_start < 0
                        or connector_end <= connector_start
                        or connector_end >= len(fill_segment.path)):
                    note_refinement("component_invalid_path_range")
                    continue
                old_connector = list(
                    fill_segment.path[connector_start:connector_end + 1])
                before_path = fill_segment.path[:connector_start + 1]
                after_path = fill_segment.path[connector_end:]
                old_crossings = int(
                    transition.get("refinement_crossings_before", 0))
                if old_crossings <= 0:
                    note_refinement("component_no_crossings")
                    continue
                if old_crossings < 6:
                    # Small isolated raster crossings are not worth another
                    # room-scale graph search.  Concentrate the offline budget
                    # on connectors that visibly cut a stack of parallel lanes.
                    note_refinement("component_low_crossing_count")
                    continue
                if id(transition) not in se2_retry_transition_ids:
                    note_refinement("component_retry_budget_skipped")
                    continue

                penalty = fixed_cleaning_penalty.copy()
                penalty |= _polylines_cleaning_mask(
                    travel_reachable.shape,
                    [before_path, after_path],
                    resolution,
                    origin_x,
                    origin_y,
                    component_penalty_width_m,
                )
                for endpoint in (
                        old_connector[0], old_connector[-1]):
                    row, col = _world_to_cell(
                        endpoint, resolution, origin_x, origin_y)
                    penalty[
                        max(0, row - component_endpoint_clear_cells):
                        min(height, row + component_endpoint_clear_cells + 1),
                        max(0, col - component_endpoint_clear_cells):
                        min(width, col + component_endpoint_clear_cells + 1),
                    ] = False

                candidate_method = "orthogonal"
                candidate = _connect_points_orthogonal_footprint_safe(
                    component_route_scope,
                    raw_free,
                    old_connector[0],
                    old_connector[-1],
                    resolution,
                    origin_x,
                    origin_y,
                    path_step_m,
                    body,
                    penalty_mask=penalty,
                )
                previous_yaw = _terminal_path_yaw(before_path)
                following_yaw = _initial_path_yaw(after_path)
                if not candidate:
                    # The orthogonal graph is deliberately conservative around
                    # raster corners.  Spend a bounded SE(2) retry only on a
                    # connector responsible for several visible crossings;
                    # low-impact connectors keep the certified original.
                    candidate_method = "se2"
                    candidate = _connect_points_footprint_safe(
                        component_route_scope,
                        turn_safe & component_route_scope,
                        raw_free,
                        old_connector[0],
                        old_connector[-1],
                        resolution,
                        origin_x,
                        origin_y,
                        path_step_m,
                        body,
                        start_yaw=previous_yaw,
                        goal_yaw=following_yaw,
                        max_se2_expansions=60_000,
                        rotation_safe_mask=rotation_safe_global,
                        stencil_table_cache=se2_stencil_table_cache,
                        penalty_mask=penalty,
                    )
                if not candidate:
                    note_refinement("component_no_candidate")
                    continue
                old_length = _partitioned_path_length(old_connector)
                candidate_length = _partitioned_path_length(candidate)
                if candidate_length > old_length + max(
                        2.0, old_length * 0.35):
                    note_refinement("component_length_regression")
                    continue
                old_turns = max(
                    0, len(_simplify_collinear(old_connector)) - 2)
                candidate_turns = max(
                    0, len(_simplify_collinear(candidate)) - 2)
                if candidate_turns > old_turns + max(
                        0, int(refine_max_extra_turns)):
                    note_refinement("component_turn_regression")
                    continue
                candidate_crossings = _proper_polyline_crossing_count_many(
                    candidate, component_crossing_references)
                if candidate_crossings >= old_crossings:
                    note_refinement("component_no_crossing_gain")
                    continue
                old_local_crossings = sum(
                    _proper_polyline_crossing_count(
                        old_connector, local_reference)
                    for local_reference in (before_path, after_path)
                    if len(local_reference) >= 2)
                candidate_local_crossings = sum(
                    _proper_polyline_crossing_count(
                        candidate, local_reference)
                    for local_reference in (before_path, after_path)
                    if len(local_reference) >= 2)
                if candidate_local_crossings > old_local_crossings:
                    note_refinement("component_local_crossing_regression")
                    continue
                old_mask = _polyline_cleaning_mask(
                    travel_reachable.shape,
                    old_connector,
                    resolution,
                    origin_x,
                    origin_y,
                    component_penalty_width_m,
                )
                candidate_mask = _polyline_cleaning_mask(
                    travel_reachable.shape,
                    candidate,
                    resolution,
                    origin_x,
                    origin_y,
                    component_penalty_width_m,
                )
                if int((candidate_mask & penalty).sum()) > int(
                        (old_mask & penalty).sum()):
                    note_refinement("component_overlap_regression")
                    continue
                candidate_start_yaw = _initial_path_yaw(candidate)
                candidate_end_yaw = _terminal_path_yaw(candidate)
                if (previous_yaw is not None
                        and candidate_start_yaw is not None
                        and not _shortest_rotation_is_free(
                            raw_free,
                            candidate[0],
                            previous_yaw,
                            candidate_start_yaw,
                            component_refine_body_samples,
                            resolution,
                            origin_x,
                            origin_y,
                        )):
                    note_refinement("component_unsafe_start_rotation")
                    continue
                if (following_yaw is not None
                        and candidate_end_yaw is not None
                        and not _shortest_rotation_is_free(
                            raw_free,
                            candidate[-1],
                            candidate_end_yaw,
                            following_yaw,
                            component_refine_body_samples,
                            resolution,
                            origin_x,
                            origin_y,
                        )):
                    note_refinement("component_unsafe_end_rotation")
                    continue

                fill_segment.path[
                    connector_start:connector_end + 1] = candidate
                path_delta = len(candidate) - len(old_connector)
                new_connector_end = connector_start + len(candidate) - 1
                for other_transition in (
                        fill_segment.component_transition_diagnostics):
                    if other_transition is transition:
                        continue
                    if int(other_transition["path_start_index"]) > connector_end:
                        other_transition["path_start_index"] = int(
                            other_transition["path_start_index"]) + path_delta
                        other_transition["path_end_index"] = int(
                            other_transition["path_end_index"]) + path_delta
                transition["path_end_index"] = int(new_connector_end)
                transition["strategy"] = (
                    f"post_refined_{candidate_method}_"
                    f"{transition['strategy']}")
                transition["length_m"] = float(candidate_length)
                transition["overlap_cells"] = int(
                    (candidate_mask & penalty).sum())
                transition["crossings_before"] = int(old_crossings)
                transition["crossings_after"] = int(candidate_crossings)
                note_refinement("component_accepted")
                refined_transfer_count += 1
                refined_transfer_crossing_reduction += (
                    old_crossings - candidate_crossings)
                component_connector_refined = True

        if component_connector_refined:
            # Connector endpoints are immutable.  Rebuild the one-shot route
            # and global segment indices after replacing variable-length slices
            # inside fill segments; hard stops are derived later from this path.
            global_path = []
            for rebuilt_segment in segments:
                if (global_path and rebuilt_segment.path
                        and _dist(
                            global_path[-1],
                            rebuilt_segment.path[0]) <= 1e-7):
                    rebuilt_segment.path_start_idx = len(global_path) - 1
                else:
                    rebuilt_segment.path_start_idx = len(global_path)
                _extend_dedup(global_path, rebuilt_segment.path)
                rebuilt_segment.path_end_idx = len(global_path) - 1

    if refine_inter_region_transfers and len(segments) > 2:
        # Inter-room transfers are initially planned before the future rooms'
        # exact fill/perimeter geometry exists.  Once every cleaning segment is
        # fixed, make one conservative post-pass over those transfers only.
        # Region order, swaths, red boundaries and their entry points remain
        # immutable; a candidate is confined to the same door/corridor graph
        # chain and must improve every local route-quality dimension.
        cleaning_penalty = np.zeros_like(travel_reachable, dtype=bool)
        penalty_width_m = max(
            resolution, min(0.12, swath_spacing_m * 0.25))
        for cleaning_segment in segments:
            if cleaning_segment.kind not in ("fill", "perimeter"):
                continue
            if (refine_fill_only_penalty
                    and cleaning_segment.kind == "perimeter"):
                continue
            cleaning_penalty |= _polyline_cleaning_mask(
                travel_reachable.shape,
                cleaning_segment.path,
                resolution,
                origin_x,
                origin_y,
                penalty_width_m,
            )
        endpoint_clear_cells = max(
            1, int(math.ceil(0.45 / resolution)))
        refine_halo_cells = max(
            1, int(math.ceil(0.25 / resolution)))
        refine_body_samples = _footprint_samples(body, resolution)

        refinement_segments = list(enumerate(segments))
        if refine_transfer_reverse:
            refinement_segments.reverse()
        for segment_index, segment in refinement_segments:
            if (segment.kind != "transfer"
                    or segment.from_region_id < 0
                    or segment.to_region_id < 0
                    or (segment.from_region_id == segment.to_region_id
                        and not refine_same_region_transfers)
                    or len(segment.path) < 2):
                continue
            if (segment_index > 0
                    and segments[segment_index - 1].continuous_to_next):
                # A one-way close-wall arc exports a tangent-continuous escape;
                # changing that prepared geometry would invalidate its body-yaw
                # certificate even when the replacement centre line looks free.
                continue
            old_crossings = sum(
                _proper_polyline_crossing_count(
                    segment.path, other_segment.path)
                for other_index, other_segment in enumerate(segments)
                if other_index != segment_index
            )
            if old_crossings <= 0:
                note_refinement("no_crossings")
                continue
            chain = (
                [segment.from_region_id]
                if segment.from_region_id == segment.to_region_id
                else shortest_region_chain(
                    segment.from_region_id, segment.to_region_id))
            route_scope = np.zeros_like(travel_reachable, dtype=bool)
            route_scope_ok = True
            if not chain:
                # A raster-thin doorway can be traversable by the exact body but
                # absent from the one-cell region adjacency graph.  The existing
                # certified connector is still an authoritative door/corridor
                # guide; search only inside a 1 m tube around it rather than
                # widening scope to the whole building.
                route_scope = _polyline_cleaning_mask(
                    travel_reachable.shape,
                    segment.path,
                    resolution,
                    origin_x,
                    origin_y,
                    2.0,
                )
                note_refinement("certified_route_scope")
            else:
                for chain_region_id in chain:
                    chain_mask = region_lookup[chain_region_id].travel_mask
                    if chain_mask is None:
                        route_scope_ok = False
                        break
                    route_scope |= chain_mask
            if not route_scope_ok:
                note_refinement("invalid_region_scope")
                continue
            route_scope = (
                dilate_binary(route_scope, refine_halo_cells)
                & travel_reachable)
            penalty = cleaning_penalty.copy()
            if refine_against_all_routes:
                for other_index, other_segment in enumerate(segments):
                    if (other_index == segment_index
                            or other_segment.kind != "transfer"):
                        continue
                    penalty |= _polyline_cleaning_mask(
                        travel_reachable.shape,
                        other_segment.path,
                        resolution,
                        origin_x,
                        origin_y,
                        penalty_width_m,
                    )
            for endpoint in (segment.path[0], segment.path[-1]):
                row, col = _world_to_cell(
                    endpoint, resolution, origin_x, origin_y)
                penalty[
                    max(0, row - endpoint_clear_cells):
                    min(height, row + endpoint_clear_cells + 1),
                    max(0, col - endpoint_clear_cells):
                    min(width, col + endpoint_clear_cells + 1),
                ] = False
            previous_boundary_yaw = (
                _terminal_path_yaw(segments[segment_index - 1].path)
                if segment_index > 0 else None)
            following_boundary_yaw = (
                _initial_path_yaw(segments[segment_index + 1].path)
                if segment_index + 1 < len(segments) else None)
            candidate = _connect_points_orthogonal_footprint_safe(
                route_scope,
                raw_free,
                segment.path[0],
                segment.path[-1],
                resolution,
                origin_x,
                origin_y,
                path_step_m,
                body,
                penalty_mask=penalty,
            )
            if not candidate:
                candidate = _connect_points_footprint_safe(
                    route_scope,
                    turn_safe & route_scope,
                    raw_free,
                    segment.path[0],
                    segment.path[-1],
                    resolution,
                    origin_x,
                    origin_y,
                    path_step_m,
                    body,
                    start_yaw=previous_boundary_yaw,
                    goal_yaw=following_boundary_yaw,
                    max_se2_expansions=90_000,
                    rotation_safe_mask=rotation_safe_global,
                    stencil_table_cache=se2_stencil_table_cache,
                    penalty_mask=penalty,
                )
            if not candidate:
                note_refinement("no_candidate")
                continue
            old_length = _partitioned_path_length(segment.path)
            candidate_length = _partitioned_path_length(candidate)
            if candidate_length > old_length + max(0.75, old_length * 0.15):
                note_refinement("length_regression")
                continue
            old_turns = max(0, len(_simplify_collinear(segment.path)) - 2)
            candidate_turns = max(0, len(_simplify_collinear(candidate)) - 2)
            if candidate_turns > old_turns + max(
                    0, int(refine_max_extra_turns)):
                note_refinement("turn_regression")
                continue
            candidate_crossings = sum(
                _proper_polyline_crossing_count(
                    candidate, other_segment.path)
                for other_index, other_segment in enumerate(segments)
                if other_index != segment_index
            )
            if candidate_crossings >= old_crossings:
                note_refinement("no_crossing_gain")
                continue
            old_mask = _polyline_cleaning_mask(
                travel_reachable.shape,
                segment.path,
                resolution,
                origin_x,
                origin_y,
                penalty_width_m,
            )
            candidate_mask = _polyline_cleaning_mask(
                travel_reachable.shape,
                candidate,
                resolution,
                origin_x,
                origin_y,
                penalty_width_m,
            )
            if int((candidate_mask & penalty).sum()) > int(
                    (old_mask & penalty).sum()):
                note_refinement("overlap_regression")
                continue
            candidate_start_yaw = _initial_path_yaw(candidate)
            candidate_end_yaw = _terminal_path_yaw(candidate)
            if segment_index > 0 and candidate_start_yaw is not None:
                previous_yaw = previous_boundary_yaw
                if (previous_yaw is not None
                        and not _shortest_rotation_is_free(
                            raw_free,
                            candidate[0],
                            previous_yaw,
                            candidate_start_yaw,
                            refine_body_samples,
                            resolution,
                            origin_x,
                            origin_y,
                        )):
                    note_refinement("unsafe_start_rotation")
                    continue
            if (segment_index + 1 < len(segments)
                    and candidate_end_yaw is not None):
                following_yaw = following_boundary_yaw
                if (following_yaw is not None
                        and not _shortest_rotation_is_free(
                            raw_free,
                            candidate[-1],
                            candidate_end_yaw,
                            following_yaw,
                            refine_body_samples,
                            resolution,
                            origin_x,
                            origin_y,
                        )):
                    note_refinement("unsafe_end_rotation")
                    continue
            segment.path = candidate
            note_refinement("accepted")
            refined_transfer_count += 1
            refined_transfer_crossing_reduction += (
                old_crossings - candidate_crossings)

        if refined_transfer_count:
            # Rebuild the one-shot polyline and all segment index metadata after
            # replacing complete connector segments.  Endpoints are immutable,
            # so the existing segment sequence and phase semantics are retained.
            global_path = []
            for segment in segments:
                if (global_path and segment.path
                        and _dist(global_path[-1], segment.path[0]) <= 1e-7):
                    segment.path_start_idx = len(global_path) - 1
                else:
                    segment.path_start_idx = len(global_path)
                _extend_dedup(global_path, segment.path)
                segment.path_end_idx = len(global_path) - 1

    if skipped_room_count:
        failures.append(f"{skipped_room_count}_narrow_regions_without_lanes")
    if not global_path:
        return empty(
            ";".join(failures) or "partitioned_path_empty",
            travel_reachable, snapped_start)

    if enable_cleaner_semantics:
        (cleaner_profile,
         cleaner_center_path,
         cleaner_semantics_valid,
         cleaner_semantics_failure_reason,
         cleaner_mode_point_counts,
         boundary_type_point_counts) = _annotate_cleaner_semantics(
            segments=segments,
            global_path=global_path,
            regions=regions,
            raw_free_mask=raw_free,
            obstacle_mask=grid > 50,
            selection_boundary_mask=selection_boundary_mask,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            footprint=body,
            max_offset_m=cleaner_max_offset_m,
            wall_gap_m=cleaner_wall_gap_m,
            transition_distance_m=cleaner_transition_distance_m,
        )
    else:
        cleaner_profile = [CleanerCommand() for _ in global_path]
        cleaner_center_path = list(global_path)
        cleaner_semantics_valid = True
        cleaner_semantics_failure_reason = ""
        cleaner_mode_point_counts = {
            CleanerMode.EDGE_CENTER.value: len(global_path)}
        boundary_type_point_counts = {
            BoundaryType.NONE.value: len(global_path)}
        for segment in segments:
            segment.cleaner_mode = CleanerMode.EDGE_CENTER
            segment.cleaner_offset_m = 0.0
            segment.cleaner_profile = [
                CleanerCommand() for _ in segment.path]
    if not cleaner_semantics_valid:
        failures.append(
            "cleaner_semantics="
            f"{cleaner_semantics_failure_reason or 'invalid'}")

    # One rolling FollowPath goal contains one cleaning lane (plus its incoming
    # yellow connector) or one region/phase transfer.  This prevents MPPI's
    # nearest-path pruning from selecting a spatially adjacent future lane and
    # makes every sharp stop/rotate corner explicit at a turn-safe endpoint.
    hard_stop_indices = set()
    corner_hard_stop_indices = set()
    hard_stop_body_samples = _footprint_samples(body, resolution)
    for segment in segments:
        if segment.path_end_idx > 0 and not segment.continuous_to_next:
            hard_stop_indices.add(segment.path_end_idx)
        if segment.kind in ("transfer", "fill"):
            # A*/portal connectors are accepted only after an explicit
            # stop-rotate-drive body sweep.  Preserve that contract at material
            # corners; otherwise RosBridge's 0.55 m lookahead begins rotating
            # before a narrow doorway and swings TD25A's rear into the jamb.
            start_index = max(1, segment.path_start_idx + 1)
            end_index = min(
                len(global_path) - 2, segment.path_end_idx - 1)
            for index in range(start_index, end_index + 1):
                incoming = (
                    global_path[index][0] - global_path[index - 1][0],
                    global_path[index][1] - global_path[index - 1][1],
                )
                outgoing = (
                    global_path[index + 1][0] - global_path[index][0],
                    global_path[index + 1][1] - global_path[index][1],
                )
                if (math.hypot(*incoming) <= 1e-7
                        or math.hypot(*outgoing) <= 1e-7):
                    continue
                incoming_yaw = math.atan2(incoming[1], incoming[0])
                outgoing_yaw = math.atan2(outgoing[1], outgoing[0])
                yaw_delta = _wrap_angle(outgoing_yaw - incoming_yaw)
                if abs(yaw_delta) < math.radians(35.0):
                    continue
                # A turn-safe centre has circular clearance for the farthest
                # body corner, so every intermediate yaw is safe by
                # construction.  This fast proof avoids re-validating a 1.3 m
                # window at thousands of ordinary BCD/room corners on large
                # maps.  Only aligned doorway/corridor corners outside that
                # conservative mask need the expensive exact-body check below.
                if _point_is_free(
                        turn_safe, global_path[index],
                        resolution, origin_x, origin_y):
                    hard_stop_indices.add(index)
                    corner_hard_stop_indices.add(index)
                    continue
                # Most simplified connectors can follow a 45/90 degree corner
                # continuously with their tangent yaws.  Add a stop only when
                # the exact body sweep of a short local window says that
                # continuous motion is unsafe; this avoids hundreds of visible
                # pauses on ordinary open-room bends.
                local_start = index
                walked = 0.0
                while local_start > segment.path_start_idx and walked < 0.65:
                    walked += _dist(
                        global_path[local_start - 1],
                        global_path[local_start])
                    local_start -= 1
                local_end = index
                walked = 0.0
                while local_end < segment.path_end_idx and walked < 0.65:
                    walked += _dist(
                        global_path[local_end],
                        global_path[local_end + 1])
                    local_end += 1
                continuous_safe, _ = validate_footprint_path(
                    raw_free,
                    global_path[local_start:local_end + 1],
                    resolution,
                    origin_x,
                    origin_y,
                    footprint=body,
                    lookahead_m=0.0,
                    first_yaw_immediate=True,
                )
                if continuous_safe:
                    continue
                if _shortest_rotation_is_free(
                        raw_free, global_path[index], incoming_yaw,
                        outgoing_yaw, hard_stop_body_samples,
                        resolution, origin_x, origin_y):
                    hard_stop_indices.add(index)
                    corner_hard_stop_indices.add(index)
        if segment.kind != "fill" or not segment.swaths:
            continue
        search_index = max(0, segment.path_start_idx)
        for _, finish in segment.swaths:
            found = next((
                index for index in range(
                    search_index, segment.path_end_idx + 1)
                if _dist(global_path[index], finish) <= 1e-5
            ), None)
            if found is None:
                continue
            hard_stop_indices.add(found)
            search_index = found
    raw_hard_stops = sorted(
        index for index in hard_stop_indices
        if 0 < index <= len(global_path) - 1)
    cumulative = [0.0]
    for first, second in zip(global_path, global_path[1:]):
        cumulative.append(cumulative[-1] + _dist(first, second))
    ordered_hard_stops: List[int] = []
    previous_index = 0
    for index in raw_hard_stops:
        # Adjacent segment boundaries can differ by one 0.10 m raster point.
        # Sending a separate action for that sliver creates a visible pause but
        # adds no safety.  Lane endpoints are at least min_swath_m apart and are
        # retained; only redundant phase/connector boundaries are coalesced.
        if (index in corner_hard_stop_indices
                or cumulative[index] - cumulative[previous_index] >= 0.25
                or index == len(global_path) - 1):
            ordered_hard_stops.append(index)
            previous_index = index

    footprint_diagnostics: Dict[str, object] = {}
    footprint_valid, footprint_violations = (
        _validate_partitioned_execution_path(
            raw_free,
            global_path,
            ordered_hard_stops,
            resolution,
            origin_x,
            origin_y,
            body,
            diagnostics=footprint_diagnostics,
        ))
    if not footprint_valid:
        failures.append(
            f"footprint_violations={footprint_violations}"
            f"(translation={footprint_diagnostics.get('translation', 0)},"
            f"rotation={footprint_diagnostics.get('rotation', 0)},"
            f"bad_slices={footprint_diagnostics.get('bad_slices', [])[:8]})")

    segment_gaps = [
        _dist(first.path[-1], second.path[0])
        for first, second in zip(segments, segments[1:])
        if first.path and second.path
    ]
    max_segment_gap = max(segment_gaps, default=0.0)
    path_continuous = max_segment_gap <= max(0.01, resolution * 0.15)
    if not path_continuous:
        failures.append(f"path_gap={max_segment_gap:.3f}m")

    arrival_yaws, departure_yaws = _explicit_path_yaw_pairs(
        global_path,
        fallback_yaw=(float(robot_yaw) if robot_yaw is not None else 0.0),
    )
    cleaning_mask = _polyline_cleaning_mask(
        (height, width), global_path,
        resolution, origin_x, origin_y, clean_width_m)
    centered_brush_mask = _forward_brush_cleaning_mask(
        (height, width), global_path, arrival_yaws, departure_yaws,
        resolution, origin_x, origin_y)
    actual_brush_mask = _forward_brush_cleaning_mask(
        (height, width), global_path, arrival_yaws, departure_yaws,
        resolution, origin_x, origin_y,
        cleaner_profile=cleaner_profile)
    coverage_target_turn_safe = turn_safe & ~discarded_small_turn_safe
    turn_safe_cells = int(coverage_target_turn_safe.sum())
    turn_safe_coverage = (
        float((cleaning_mask & coverage_target_turn_safe).sum())
        / turn_safe_cells
        if turn_safe_cells else 0.0
    )
    reachable_target = travel_reachable & mission_mask
    reachable_cells = int(reachable_target.sum())
    reachable_coverage = (
        float((cleaning_mask & reachable_target).sum()) / reachable_cells
        if reachable_cells else 0.0
    )
    # The half-width reachable mask is intentionally broad enough to preserve
    # door topology.  It can also contain aligned-only dead ends and lidar
    # slivers where the 1.05 m body cannot turn or leave.  Keep that ratio as a
    # strict diagnostic; use brush reach from full-body turn-safe poses as the
    # executable coverage target.
    serviceable_target = (
        _disk_dilate_mask(
            coverage_target_turn_safe,
            resolution,
            max(0.0, clean_width_m * 0.5),
        )
        & reachable_target
    )
    discarded_small_target = (
        _disk_dilate_mask(
            discarded_small_turn_safe,
            resolution,
            max(0.0, clean_width_m * 0.5),
        )
        & reachable_target
        & ~serviceable_target
    )
    discarded_small_area_m2 = (
        float(discarded_small_target.sum()) * resolution ** 2)
    serviceable_cells = int(serviceable_target.sum())
    serviceable_coverage = (
        float((cleaning_mask & serviceable_target).sum()) / serviceable_cells
        if serviceable_cells else 0.0
    )
    actual_brush_coverage = (
        float((actual_brush_mask & serviceable_target).sum())
        / serviceable_cells
        if serviceable_cells else 0.0
    )
    centered_brush_coverage = (
        float((centered_brush_mask & serviceable_target).sum())
        / serviceable_cells
        if serviceable_cells else 0.0
    )
    cleaner_extension_gain_area_m2 = (
        float((actual_brush_mask & serviceable_target
               & ~centered_brush_mask).sum()) * resolution ** 2)
    region_coverage: Dict[int, float] = {}
    region_serviceable_coverage: Dict[int, float] = {}
    region_actual_brush_coverage: Dict[int, float] = {}
    for region in regions:
        target = (
            region.travel_mask & mission_mask
            if region.travel_mask is not None else region.mask
        )
        target_cells = int(target.sum())
        region_coverage[region.region_id] = (
            float((cleaning_mask & target).sum()) / target_cells
            if target_cells else 0.0
        )
        region_serviceable_target = serviceable_target & target
        region_serviceable_cells = int(region_serviceable_target.sum())
        region_serviceable_coverage[region.region_id] = (
            float((cleaning_mask & region_serviceable_target).sum())
            / region_serviceable_cells
            if region_serviceable_cells else 0.0
        )
        region_actual_brush_coverage[region.region_id] = (
            float((actual_brush_mask & region_serviceable_target).sum())
            / region_serviceable_cells
            if region_serviceable_cells else 0.0
        )
    for region_id, (completed, missing_mandatory,
                    optional_missing) in perimeter_issues.items():
        # At least one real edge pass remains mandatory.  Once it has run, an
        # inaccessible additional contour is a map-noise/completion diagnostic;
        # the global and per-component serviceable coverage gates still catch
        # any real uncovered floor.
        if completed == 0:
            failures.append(
                f"region_{region_id}_perimeter_missing="
                f"{max(1, missing_mandatory)}")
        else:
            optional_missing += missing_mandatory
        if optional_missing:
            failures.append(
                f"region_{region_id}_optional_completion_skipped="
                f"{optional_missing}")
    for region_id, issues in component_issues.items():
        if region_serviceable_coverage.get(region_id, 0.0) < 0.95:
            failures.append(
                f"region_{region_id}_component_completion_failed="
                f"{','.join(issues)}")
        else:
            # The skipped BCD lane is geometrically redundant: neighbouring
            # lanes/brush reach already cover the region above the mission
            # threshold.  Keeping the impossible connector would add a tangle
            # without cleaning any required floor.
            failures.append(
                f"region_{region_id}_redundant_components_skipped="
                f"{len(issues)}")
    if turn_safe_coverage < 0.98:
        failures.append(f"turn_safe_coverage={turn_safe_coverage:.3f}")
    if serviceable_coverage < 0.95:
        failures.append(f"serviceable_coverage={serviceable_coverage:.3f}")
    if actual_brush_coverage < 0.95:
        failures.append(
            f"actual_brush_coverage={actual_brush_coverage:.3f}")
    if reachable_coverage < 0.95:
        failures.append(
            f"strict_reachable_coverage={reachable_coverage:.3f}")

    return PartitionedCoveragePlan(
        path=global_path,
        segments=segments,
        regions=regions,
        visit_order=visit_order,
        free_mask=travel_reachable,
        snapped_start=snapped_start,
        swaths=all_swaths,
        hard_stop_indices=ordered_hard_stops,
        failure_reason=";".join(failures),
        footprint_valid=footprint_valid,
        footprint_violation_count=footprint_violations,
        path_continuous=path_continuous,
        max_segment_gap_m=max_segment_gap,
        turn_safe_coverage_ratio=turn_safe_coverage,
        serviceable_coverage_ratio=serviceable_coverage,
        actual_brush_coverage_ratio=actual_brush_coverage,
        reachable_coverage_ratio=reachable_coverage,
        region_coverage_ratios=region_coverage,
        region_serviceable_coverage_ratios=region_serviceable_coverage,
        region_actual_brush_coverage_ratios=region_actual_brush_coverage,
        selection_mode=(
            f"{region_order_mode}_exit_aware"
            if exit_aware_enabled and region_order_mode != "station"
            else "exit_aware" if exit_aware_enabled
            else region_order_mode if region_order_mode != "station"
            else "baseline"),
        exit_aware_candidate_region_ids=sorted(
            exit_aware_candidate_regions),
        current_first_candidate_recommended=current_first_recommended,
        graph_order_candidate_recommended=graph_order_recommended,
        arrival_yaws=arrival_yaws,
        departure_yaws=departure_yaws,
        cleaner_profile=cleaner_profile,
        cleaner_center_path=cleaner_center_path,
        centered_brush_coverage_ratio=centered_brush_coverage,
        cleaner_extension_gain_area_m2=cleaner_extension_gain_area_m2,
        cleaner_semantics_valid=cleaner_semantics_valid,
        cleaner_semantics_failure_reason=(
            cleaner_semantics_failure_reason),
        cleaner_mode_point_counts=cleaner_mode_point_counts,
        boundary_type_point_counts=boundary_type_point_counts,
        cleaner_max_offset_m=cleaner_max_offset_m,
        raw_free_mask=raw_free,
        serviceable_target_mask=serviceable_target,
        discarded_small_component_count=discarded_small_component_count,
        discarded_small_area_m2=discarded_small_area_m2,
        refined_transfer_count=refined_transfer_count,
        refined_transfer_crossing_reduction=(
            refined_transfer_crossing_reduction),
        refinement_diagnostics=refinement_diagnostics,
        footprint=body,
        coverage_complete=(
            len(visit_order) == len(regions)
            and footprint_valid
            and path_continuous
            and cleaner_semantics_valid
            and turn_safe_coverage >= 0.98
            and serviceable_coverage >= 0.95
            and actual_brush_coverage >= 0.95
            and not any(
                "unreachable" in failure
                or "component_completion_failed" in failure
                or "perimeter_missing" in failure
                for failure in failures)
        ),
    )


def _partitioned_path_length(path: Sequence[Point]) -> float:
    return sum(_dist(first, second) for first, second in zip(path, path[1:]))


def _proper_polyline_crossing_count(
    first_path: Sequence[Point],
    second_path: Sequence[Point],
) -> int:
    """Count strict interior intersections between two centre polylines.

    Shared endpoints and collinear reuse are deliberately excluded.  The
    definition matches the operator-facing route-quality metric: a doorway may
    have to be reused, while a connector that cuts through a cleaning lane is a
    visible proper crossing that can sometimes be removed.
    """
    first_simplified = _simplify_collinear(first_path)
    second_simplified = _simplify_collinear(second_path)
    first_edges = list(zip(first_simplified, first_simplified[1:]))
    second_edges = list(zip(second_simplified, second_simplified[1:]))
    return _proper_edge_crossing_count(first_edges, second_edges)


def _proper_edge_crossing_count(
    first_edges: Sequence[Tuple[Point, Point]],
    second_edges: Sequence[Tuple[Point, Point]],
) -> int:
    """Count strict crossings for already simplified edge lists."""

    def orientation(a: Point, b: Point, c: Point) -> float:
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    crossings = 0
    for a, b in first_edges:
        if _dist(a, b) <= 1e-9:
            continue
        min_ax, max_ax = sorted((a[0], b[0]))
        min_ay, max_ay = sorted((a[1], b[1]))
        for c, d in second_edges:
            if _dist(c, d) <= 1e-9:
                continue
            min_cx, max_cx = sorted((c[0], d[0]))
            min_cy, max_cy = sorted((c[1], d[1]))
            if (max_ax <= min_cx + 1e-8
                    or max_cx <= min_ax + 1e-8
                    or max_ay <= min_cy + 1e-8
                    or max_cy <= min_ay + 1e-8):
                continue
            if (orientation(a, b, c) * orientation(a, b, d) < -1e-10
                    and orientation(c, d, a) * orientation(c, d, b)
                    < -1e-10):
                crossings += 1
    return crossings


def _proper_polyline_crossing_count_many(
    first_path: Sequence[Point],
    second_paths: Sequence[Sequence[Point]],
) -> int:
    """Count crossings against many paths while simplifying the first once."""
    first_simplified = _simplify_collinear(first_path)
    first_edges = list(zip(first_simplified, first_simplified[1:]))
    crossings = 0
    for second_path in second_paths:
        second_simplified = _simplify_collinear(second_path)
        second_edges = list(zip(
            second_simplified, second_simplified[1:]))
        crossings += _proper_edge_crossing_count(
            first_edges, second_edges)
    return crossings


def _canonical_swath_multiset(swaths: Sequence[Swath]) -> List[Tuple[int, ...]]:
    """Return direction-independent, deterministic swath geometry.

    Exit-aware planning may reverse a lane, but it must never remove, shorten,
    or invent one.  Quantising below one micrometre avoids floating-point
    representation noise while remaining far tighter than the occupancy-grid
    resolution.
    """
    scale = 1_000_000.0
    canonical: List[Tuple[int, ...]] = []
    for first, second in swaths:
        a = (int(round(first[0] * scale)), int(round(first[1] * scale)))
        b = (int(round(second[0] * scale)), int(round(second[1] * scale)))
        low, high = sorted((a, b))
        canonical.append((low[0], low[1], high[0], high[1]))
    return sorted(canonical)


def _partitioned_route_quality(
    plan: PartitionedCoveragePlan,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> Dict[str, float]:
    """Measure centre-line retrace and proper crossings for candidate gating.

    The cleaning brush intentionally overlaps adjacent lanes and a room's red
    perimeter necessarily touches its own yellow lane endpoints.  Neither is a
    route-planning defect, so this metric samples only the centre line and
    excludes same-region ``fill -> perimeter`` contact.  It remains an upper
    bound because a dead-end room physically has to reuse its only doorway.
    """
    sample_step = max(0.02, min(0.05, resolution * 0.5))
    seen_cells: Dict[Tuple[int, int], Tuple[str, int]] = {}
    last_cell: Optional[Tuple[int, int]] = None
    samples = 0
    avoidable_repeats = 0
    for segment in plan.segments:
        current = (segment.kind, int(segment.region_id))
        for first, second in zip(segment.path, segment.path[1:]):
            distance = _dist(first, second)
            count = max(1, int(math.ceil(distance / sample_step)))
            for sample_index in range(count):
                ratio = sample_index / count
                point = (
                    first[0] + (second[0] - first[0]) * ratio,
                    first[1] + (second[1] - first[1]) * ratio,
                )
                cell = (
                    int(math.floor((point[1] - origin_y) / resolution)),
                    int(math.floor((point[0] - origin_x) / resolution)),
                )
                if cell == last_cell:
                    continue
                previous = seen_cells.get(cell)
                if previous is not None and not (
                        previous[0] == "fill"
                        and current[0] == "perimeter"
                        and previous[1] == current[1]):
                    avoidable_repeats += 1
                elif previous is None:
                    seen_cells[cell] = current
                samples += 1
                last_cell = cell

    # Label every simplified straight edge, then use spatial buckets so large
    # maps do not pay the benchmark helper's quadratic all-pairs cost.
    labelled: List[
        Tuple[Point, Point, str, int, int, int]
    ] = []
    for segment_index, segment in enumerate(plan.segments):
        simplified = _simplify_collinear(segment.path)
        for local_index, (first, second) in enumerate(
                zip(simplified, simplified[1:])):
            if _dist(first, second) <= 1e-9:
                continue
            labelled.append((
                first, second, segment.kind, int(segment.region_id),
                segment_index, local_index,
            ))

    bucket_size = max(0.50, resolution * 5.0)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    candidate_pairs = set()
    for line_index, (first, second, _, _, _, _) in enumerate(labelled):
        min_x, max_x = sorted((first[0], second[0]))
        min_y, max_y = sorted((first[1], second[1]))
        col0 = int(math.floor((min_x - origin_x) / bucket_size))
        col1 = int(math.floor((max_x - origin_x) / bucket_size))
        row0 = int(math.floor((min_y - origin_y) / bucket_size))
        row1 = int(math.floor((max_y - origin_y) / bucket_size))
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                bucket = buckets.setdefault((row, col), [])
                for previous_index in bucket:
                    candidate_pairs.add((previous_index, line_index))
                bucket.append(line_index)

    def orientation(a: Point, b: Point, c: Point) -> float:
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    avoidable_crossings = 0
    for first_index, second_index in candidate_pairs:
        (a, b, first_kind, first_region,
         first_segment, first_local) = labelled[first_index]
        (c, d, second_kind, second_region,
         second_segment, second_local) = labelled[second_index]
        if (first_segment == second_segment
                and abs(first_local - second_local) <= 1):
            continue
        min_ax, max_ax = sorted((a[0], b[0]))
        min_ay, max_ay = sorted((a[1], b[1]))
        if (max_ax <= min(c[0], d[0]) + 1e-8
                or max(c[0], d[0]) <= min_ax + 1e-8
                or max_ay <= min(c[1], d[1]) + 1e-8
                or max(c[1], d[1]) <= min_ay + 1e-8):
            continue
        if not (orientation(a, b, c) * orientation(a, b, d) < -1e-10
                and orientation(c, d, a) * orientation(c, d, b) < -1e-10):
            continue
        if ({first_kind, second_kind} == {"fill", "perimeter"}
                and first_region == second_region):
            continue
        avoidable_crossings += 1

    transfer_lengths = [
        _partitioned_path_length(segment.path)
        for segment in plan.segments if segment.kind == "transfer"
    ]
    swath_lengths = [
        _dist(start, finish) for start, finish in plan.swaths]
    path_length = _partitioned_path_length(plan.path)
    straight_lane_length = float(sum(swath_lengths))
    return {
        "avoidable_repeat_ratio": (
            avoidable_repeats / max(1, samples)),
        "avoidable_repeat_samples": float(avoidable_repeats),
        "centerline_samples": float(samples),
        "avoidable_crossings": float(avoidable_crossings),
        "transfer_length_m": float(sum(transfer_lengths)),
        "long_transfer_count": float(sum(
            length > 2.0 for length in transfer_lengths)),
        "max_transfer_length_m": float(max(transfer_lengths, default=0.0)),
        "path_length_m": path_length,
        "hard_stop_count": float(len(plan.hard_stop_indices)),
        # Operator preference: maximise long, uninterrupted cleaning runs.
        # The numerator is certified straight swath geometry; the denominator
        # includes every doorway/turn/transfer metre needed to execute it.
        "straight_lane_length_m": straight_lane_length,
        "straight_lane_ratio": (
            straight_lane_length / path_length if path_length > 1e-9 else 0.0),
        "mean_swath_length_m": (
            straight_lane_length / len(swath_lengths)
            if swath_lengths else 0.0),
        "short_swath_count": float(sum(
            length < 2.0 for length in swath_lengths)),
    }


def _canonical_perimeter_geometry(
    plan: PartitionedCoveragePlan,
) -> Dict[int, set]:
    """Return actual red-path geometry per region, independent of direction."""
    geometry: Dict[int, set] = {}
    for segment in plan.segments:
        if segment.kind != "perimeter":
            continue
        points = geometry.setdefault(int(segment.region_id), set())
        points.update(
            (int(round(float(x) * 10_000.0)),
             int(round(float(y) * 10_000.0)))
            for x, y in segment.path
        )
    return geometry


def _exit_candidate_preserves_baseline(
    baseline: PartitionedCoveragePlan,
    candidate: PartitionedCoveragePlan,
) -> bool:
    """Hard gate: route quality can never buy away safety or coverage."""
    if not (candidate.coverage_complete
            and candidate.footprint_valid
            and candidate.path_continuous
            and candidate.footprint_violation_count == 0):
        return False
    if (len(candidate.visit_order) != len(baseline.visit_order)
            or len(set(candidate.visit_order)) != len(candidate.visit_order)
            or set(candidate.visit_order) != set(baseline.visit_order)):
        return False
    if len(candidate.regions) != len(baseline.regions):
        return False
    if _canonical_swath_multiset(candidate.swaths) != _canonical_swath_multiset(
            baseline.swaths):
        return False
    baseline_perimeters = _canonical_perimeter_geometry(baseline)
    candidate_perimeters = _canonical_perimeter_geometry(candidate)
    if any(not points.issubset(candidate_perimeters.get(region_id, set()))
           for region_id, points in baseline_perimeters.items()):
        return False
    # Connector geometry can cover a few different raster cells even when the
    # complete swath and perimeter sets are identical.  Treat at most half a
    # tenth of a percentage point as rasterisation noise, while retaining the absolute
    # mission gates (98% turn-safe and 95% both geometrically serviceable and
    # reachable by the measured forward brush).
    # Body collision and continuity remain exact zero-tolerance invariants.
    tolerance = 0.001
    if (candidate.turn_safe_coverage_ratio + tolerance
            < baseline.turn_safe_coverage_ratio
            or candidate.serviceable_coverage_ratio + tolerance
            < baseline.serviceable_coverage_ratio
            or candidate.actual_brush_coverage_ratio + tolerance
            < baseline.actual_brush_coverage_ratio
            or candidate.reachable_coverage_ratio + tolerance
            < baseline.reachable_coverage_ratio):
        return False
    if (candidate.turn_safe_coverage_ratio < 0.98
            or candidate.serviceable_coverage_ratio < 0.95
            or candidate.actual_brush_coverage_ratio < 0.95):
        return False
    for region_id, baseline_ratio in (
            baseline.region_serviceable_coverage_ratios.items()):
        candidate_ratio = candidate.region_serviceable_coverage_ratios.get(
            region_id, 0.0)
        # Planner regions can include small corridor/junction fragments rather
        # than semantic rooms.  A candidate may not worsen any such fragment;
        # when the baseline already clears 95%, it must retain that absolute
        # gate.  Global measured-brush coverage remains an unconditional 95%.
        if ((baseline_ratio >= 0.95 and candidate_ratio < 0.95)
                or (baseline_ratio < 0.95
                    and candidate_ratio + tolerance < baseline_ratio)):
            return False
        baseline_actual = baseline.region_actual_brush_coverage_ratios.get(
            region_id, 0.0)
        candidate_actual = candidate.region_actual_brush_coverage_ratios.get(
            region_id, 0.0)
        if ((baseline_actual >= 0.95 and candidate_actual < 0.95)
                or (baseline_actual < 0.95
                    and candidate_actual + tolerance < baseline_actual)):
            return False
    return True


def _strictly_prefer_exit_candidate(
    baseline_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
) -> bool:
    """Conservative multi-objective gate with a meaningful-gain threshold.

    Normal selection is Pareto-only.  One deliberately narrow exception lets
    at least five proper crossings disappear in exchange for at most 0.1
    percentage point / five centreline samples of retrace, provided the route
    also becomes at least as straight and no longer.  This prevents a tiny
    raster overlap from vetoing a visibly cleaner door/corridor connection.
    """
    epsilon = 1e-9
    crossing_gain = (
        baseline_metrics["avoidable_crossings"]
        - candidate_metrics["avoidable_crossings"])
    repeat_gain = (
        baseline_metrics["avoidable_repeat_ratio"]
        - candidate_metrics["avoidable_repeat_ratio"])
    repeat_sample_gain = (
        baseline_metrics["avoidable_repeat_samples"]
        - candidate_metrics["avoidable_repeat_samples"])
    straight_ratio_gain = (
        candidate_metrics.get("straight_lane_ratio", 0.0)
        - baseline_metrics.get("straight_lane_ratio", 0.0))
    bounded_crossing_trade = bool(
        crossing_gain >= 5.0
        and -repeat_gain <= 0.001 + epsilon
        and -repeat_sample_gain <= 5.0 + epsilon
        and straight_ratio_gain >= -epsilon
        and candidate_metrics["path_length_m"]
        <= baseline_metrics["path_length_m"] + 0.05)
    if crossing_gain < -epsilon:
        return False
    if (candidate_metrics["avoidable_repeat_ratio"]
            > baseline_metrics["avoidable_repeat_ratio"] + epsilon
            and not bounded_crossing_trade):
        return False
    if (candidate_metrics["avoidable_repeat_samples"]
            > baseline_metrics["avoidable_repeat_samples"] + epsilon
            and not bounded_crossing_trade):
        return False
    if (candidate_metrics["hard_stop_count"]
            > baseline_metrics["hard_stop_count"] + epsilon):
        return False
    if (candidate_metrics.get("straight_lane_ratio", 0.0) + epsilon
            < baseline_metrics.get("straight_lane_ratio", 0.0)):
        return False
    if (candidate_metrics.get("short_swath_count", 0.0)
            > baseline_metrics.get("short_swath_count", 0.0) + epsilon):
        return False
    if (candidate_metrics["long_transfer_count"]
            > baseline_metrics["long_transfer_count"] + epsilon):
        return False
    if (candidate_metrics["max_transfer_length_m"]
            > baseline_metrics["max_transfer_length_m"] + 0.05
            and not (
                bounded_crossing_trade
                and candidate_metrics["max_transfer_length_m"]
                <= baseline_metrics["max_transfer_length_m"] + 0.50)):
        return False
    if (candidate_metrics["path_length_m"]
            > baseline_metrics["path_length_m"] * 1.01 + 0.05):
        return False
    transfer_gain = (
        baseline_metrics["transfer_length_m"]
        - candidate_metrics["transfer_length_m"])
    meaningful_repeat = max(
        0.001,
        baseline_metrics["avoidable_repeat_ratio"] * 0.03,
    )
    return bool(
        crossing_gain >= 1.0
        or repeat_gain >= meaningful_repeat
        or straight_ratio_gain >= 0.002
        or bounded_crossing_trade
        or (transfer_gain >= 0.50
            and crossing_gain >= -epsilon
            and repeat_gain >= -epsilon)
    )


def plan_partitioned_coverage(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_world: Point,
    swath_spacing_m: float,
    robot_yaw: Optional[float] = None,
    footprint: Optional[CoverageFootprint] = None,
    clip_polygon: Optional[Sequence[Point]] = None,
    selection_boundary_polygon: Optional[Sequence[Point]] = None,
    blocked_polygons: Optional[Sequence[Sequence[Point]]] = None,
    known_free_mask: Optional[np.ndarray] = None,
    path_step_m: float = 0.10,
    min_swath_m: float = 0.45,
    min_region_area_m2: float = 3.0,
    min_cleanable_component_area_m2: float = 0.80,
    min_useful_swath_m: float = 1.30,
    min_useful_region_area_m2: float = 0.0,
    min_useful_region_lane_m: float = 0.0,
    min_fragment_cell_lane_m: float = 0.0,
    adaptive_fragment_pruning: bool = False,
    max_regions: int = 16,
    clean_width_m: float = 0.70,
    enable_cleaner_semantics: bool = True,
    cleaner_max_offset_m: float = 0.25,
    cleaner_wall_gap_m: float = 0.03,
    cleaner_transition_distance_m: float = 0.45,
    selection_policy: str = "strict",
) -> PartitionedCoveragePlan:
    """Return the proven baseline or a strictly better exit-aware candidate.

    ``baseline`` performs one legacy plan. ``shadow`` evaluates both but still
    returns the legacy geometry. The default ``strict`` policy selects the
    alternative only after the hard invariants and Pareto quality gate pass.
    ``sparse_graph`` is the operator-approved room-graph route: it starts at
    the robot, avoids completed/future room geometry and refines only certified
    inter-room transfers. None of these modes starts ROS or robot motion;
    planning remains offline.
    """
    if selection_policy not in {
            "baseline", "shadow", "strict", "sparse_graph"}:
        raise ValueError(
            "selection_policy must be baseline, shadow, strict, or sparse_graph")
    common = dict(
        data=data, width=width, height=height, resolution=resolution,
        origin_x=origin_x, origin_y=origin_y, robot_world=robot_world,
        swath_spacing_m=swath_spacing_m, robot_yaw=robot_yaw,
        footprint=footprint, clip_polygon=clip_polygon,
        selection_boundary_polygon=selection_boundary_polygon,
        blocked_polygons=blocked_polygons,
        known_free_mask=known_free_mask, path_step_m=path_step_m,
        min_swath_m=min_swath_m, min_region_area_m2=min_region_area_m2,
        min_cleanable_component_area_m2=min_cleanable_component_area_m2,
        min_useful_swath_m=min_useful_swath_m,
        min_useful_region_area_m2=min_useful_region_area_m2,
        min_useful_region_lane_m=min_useful_region_lane_m,
        min_fragment_cell_lane_m=min_fragment_cell_lane_m,
        adaptive_fragment_pruning=adaptive_fragment_pruning,
        max_regions=max_regions, clean_width_m=clean_width_m,
        enable_cleaner_semantics=enable_cleaner_semantics,
        cleaner_max_offset_m=cleaner_max_offset_m,
        cleaner_wall_gap_m=cleaner_wall_gap_m,
        cleaner_transition_distance_m=cleaner_transition_distance_m,
    )
    if selection_policy == "sparse_graph":
        # The operator-approved sparse policy may discard low-value raster
        # fingers.  Keep strict/baseline callers byte-for-byte compatible;
        # only this policy enables the per-component fragmentation floor.
        common["adaptive_fragment_pruning"] = True
        common["min_fragment_cell_lane_m"] = max(
            float(min_fragment_cell_lane_m), 5.0)
        profiles: List[Tuple[str, Dict[str, object]]] = [
            ("sparse_graph", dict(common)),
        ]
        region_relaxed = dict(common)
        region_relaxed["min_useful_region_area_m2"] = 0.0
        region_relaxed["min_useful_region_lane_m"] = 0.0
        profiles.append(("sparse_graph_region_relaxed", region_relaxed))
        lane_relaxed = dict(region_relaxed)
        lane_relaxed["min_swath_m"] = min(float(min_swath_m), 0.80)
        lane_relaxed["adaptive_fragment_pruning"] = False
        profiles.append(("sparse_graph_lane_relaxed", lane_relaxed))
        safe_fallback = dict(region_relaxed)
        safe_fallback["min_swath_m"] = min(float(min_swath_m), 0.45)
        safe_fallback["adaptive_fragment_pruning"] = False
        profiles.append(("sparse_graph_safe_fallback", safe_fallback))

        attempts: List[PartitionedCoveragePlan] = []
        for profile_name, profile in profiles:
            profile_candidates: List[PartitionedCoveragePlan] = []
            for order_mode in ("current_graph", "station_graph"):
                selected = _plan_partitioned_coverage_once(
                    **profile,
                    exit_aware_enabled=True,
                    region_order_mode=order_mode,
                    avoid_completed_route_transfers=True,
                    hard_avoid_completed_route_transfers=True,
                    avoid_future_component_swaths=True,
                    avoid_future_region_swaths=True,
                    refine_inter_region_transfers=True,
                    refine_against_all_routes=True,
                    refine_transfer_reverse=True,
                    refine_same_region_transfers=True,
                    refine_max_extra_turns=2,
                )
                selected.quality_metrics = _partitioned_route_quality(
                    selected, resolution, origin_x, origin_y)
                selected.selection_mode = f"{profile_name}_{order_mode}"
                profile_candidates.append(selected)
            attempts.extend(profile_candidates)
            valid_candidates = [
                candidate for candidate in profile_candidates
                if (candidate.coverage_complete
                    and candidate.footprint_valid
                    and candidate.footprint_violation_count == 0
                    and candidate.path_continuous
                    and candidate.actual_brush_coverage_ratio >= 0.95)
            ]
            if valid_candidates:
                return min(
                    valid_candidates,
                    key=lambda candidate: (
                        candidate.quality_metrics.get(
                            "avoidable_crossings", float("inf")),
                        candidate.quality_metrics.get(
                            "avoidable_repeat_ratio", float("inf")),
                        candidate.quality_metrics.get(
                            "avoidable_repeat_samples", float("inf")),
                        candidate.quality_metrics.get(
                            "transfer_length_m", float("inf")),
                        candidate.quality_metrics.get(
                            "path_length_m", float("inf")),
                    ),
                )
        # Preserve the most useful diagnostic if every adaptive level fails.
        return max(
            attempts,
            key=lambda plan: (
                bool(plan.path),
                plan.actual_brush_coverage_ratio,
                -plan.footprint_violation_count,
            ),
        )

    baseline = _plan_partitioned_coverage_once(
        **common, exit_aware_enabled=False)
    baseline.quality_metrics = _partitioned_route_quality(
        baseline, resolution, origin_x, origin_y)

    def with_safe_min_swath_fallback(
            selected: PartitionedCoveragePlan) -> PartitionedCoveragePlan:
        """Retry the proven 0.45 m lane floor if short fragments are unsafe.

        A 0.35 m minimum recovers useful short wall-side lanes on saved maps
        and materially reduces connector tangles.  Some clutter geometries,
        however, expose a short BCD fragment whose connector cannot sweep the
        complete TD25A body.  Never return that candidate merely for coverage:
        rerun the same public policy at 0.45 m and use it only when its full
        safety/continuity/coverage contract passes.
        """
        if (min_swath_m >= 0.45 - 1e-9
                or (selected.coverage_complete
                    and selected.footprint_valid
                    and selected.footprint_violation_count == 0
                    and selected.path_continuous)):
            return selected
        fallback_common = dict(common)
        fallback_common["min_swath_m"] = 0.45
        fallback = plan_partitioned_coverage(
            **fallback_common, selection_policy=selection_policy)
        if (fallback.coverage_complete
                and fallback.footprint_valid
                and fallback.footprint_violation_count == 0
                and fallback.path_continuous):
            fallback.alternative_quality_metrics = dict(
                selected.quality_metrics)
            fallback.selection_mode += "_min_swath_fallback"
            return fallback
        return selected

    if selection_policy == "baseline" or not baseline.coverage_complete:
        return with_safe_min_swath_fallback(baseline)

    candidate_specs: List[Tuple[str, bool, str, bool, bool]] = []
    has_exit_candidate = bool(baseline.exit_aware_candidate_region_ids)
    if has_exit_candidate:
        candidate_specs.append((
            "exit_aware_strict", True, "station", False, False))
    if baseline.current_first_candidate_recommended:
        # This is the operator's preferred default: finish the region under
        # the robot before moving on.  Combine it with the leaf-exit hint when
        # available so a one-door room still finishes at its doorway.
        candidate_specs.append((
            "current_first_strict", has_exit_candidate,
            "current_first", False, False))
    if len(baseline.regions) > 1:
        candidate_specs.append((
            "completed_and_future_route_avoidance_strict",
            has_exit_candidate,
            ("current_first"
             if baseline.current_first_candidate_recommended
             else "station"),
            True,
            True,
        ))
    if not candidate_specs:
        return with_safe_min_swath_fallback(baseline)

    candidates: List[Tuple[str, PartitionedCoveragePlan]] = []
    for (selection_name, exit_aware, order_mode,
         avoid_completed, avoid_future) in candidate_specs:
        candidate = _plan_partitioned_coverage_once(
            **common,
            exit_aware_enabled=exit_aware,
            region_order_mode=order_mode,
            avoid_completed_route_transfers=avoid_completed,
            avoid_future_component_swaths=avoid_future,
        )
        candidate.quality_metrics = _partitioned_route_quality(
            candidate, resolution, origin_x, origin_y)
        candidate.alternative_quality_metrics = dict(
            baseline.quality_metrics)
        candidates.append((selection_name, candidate))

    best_alternative = min(
        (candidate for _, candidate in candidates),
        key=lambda plan: (
            plan.quality_metrics.get("avoidable_crossings", float("inf")),
            plan.quality_metrics.get(
                "avoidable_repeat_ratio", float("inf")),
            plan.quality_metrics.get(
                "avoidable_repeat_samples", float("inf")),
            plan.quality_metrics.get("transfer_length_m", float("inf")),
            plan.quality_metrics.get("path_length_m", float("inf")),
        ),
    )
    baseline.alternative_quality_metrics = dict(
        best_alternative.quality_metrics)
    if selection_policy == "shadow":
        baseline.selection_mode = "baseline_shadow"
        return with_safe_min_swath_fallback(baseline)

    selected = baseline
    for selection_name, candidate in candidates:
        if (_exit_candidate_preserves_baseline(baseline, candidate)
                and _strictly_prefer_exit_candidate(
                    selected.quality_metrics, candidate.quality_metrics)):
            selected = candidate
            selected.selection_mode = selection_name
    if selected is not baseline:
        return with_safe_min_swath_fallback(selected)
    baseline.selection_mode = "baseline_strict_gate"
    return with_safe_min_swath_fallback(baseline)


def plan_lawnmower_coverage(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_radius_m: float,
    swath_spacing_m: float,
    robot_world: Point,
    robot_yaw: Optional[float] = None,
    clip_polygon: Optional[Sequence[Point]] = None,
    blocked_polygons: Optional[Sequence[Sequence[Point]]] = None,
    travel_clip_polygon: Optional[Sequence[Point]] = None,
    travel_radius_m: Optional[float] = None,
    axis: str = "x",
    min_swath_m: float = 0.45,
    path_step_m: float = 0.10,
    turn_radius_m: float = 0.50,   # 0.30→0.50: 掉头/换行圆角更缓, MPPI 少硬转→转弯更丝滑(受段长45%上限保护)
    forced_angle_rad: Optional[float] = None,  # [2012夜] 指定车道方向(弧度), 覆盖PCA/axis
                                               # 自动选择 —— 补漏/恢复碎块跟随主任务方向
    rect_cells: bool = False,  # [2026-07-07 弃用] BCD分解实测碎片化(9碎单元含0宽退化)+同轴合并
                               # 造发卡锐角(整晚whack-a-mole); 已改用 no_rotate 强制轴对齐替代。
    no_rotate: bool = False,   # [2026-07-07 正解] True=禁PCA旋转, 用x/y轴对齐弓字(横平竖直不斜)。
                               # 实测8起步位全零锐角(不走BCD, 整块弓字本就干净)。
) -> CoveragePlan:
    """Plan the interior boustrophedon coverage path (含 headland 内缩使弓字整体离墙).

    沿边收尾(perimeter)已拆成独立的 plan_perimeter_coverage(), 由调用方在【所有内部覆盖
    (含补扫/恢复)全部跑完后】作为最终收尾阶段单独生成执行 —— 这样沿边不会被中途恢复/补扫
    重规划丢弃(之前 append 在同一条 path 里, 内部一 abort 走恢复就把沿边连累丢了)。
    """
    coverage_free = build_coverage_free_mask(
        data=data,
        width=width,
        height=height,
        resolution=resolution,
        robot_radius_m=robot_radius_m,
        clip_polygon=clip_polygon,
        blocked_polygons=blocked_polygons,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    travel_clearance = (robot_radius_m if travel_radius_m is None
                        else max(0.0, float(travel_radius_m)))
    travel_free = build_coverage_free_mask(
        data=data,
        width=width,
        height=height,
        resolution=resolution,
        robot_radius_m=travel_clearance,
        clip_polygon=travel_clip_polygon,
        blocked_polygons=blocked_polygons,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    # First establish the physical-footprint travel component from the robot.
    # Fill-safe cells may form several islands separated by a doorway/corridor
    # that the robot can traverse but where we intentionally do not place a
    # cleaning swath.  Keep every fill island inside this travel component.
    travel_start_cell = _snap_to_free(travel_free, _world_to_cell(
        robot_world, resolution, origin_x, origin_y))
    if travel_start_cell is None:
        return CoveragePlan([], [], coverage_free, robot_world)
    travel_reachable = largest_component(travel_free, seed=travel_start_cell)
    reachable = coverage_free & travel_reachable

    # For a user-selected zone the robot may be outside the fill mask.  Prefer
    # every fill island reachable through the chassis-clearance component.  If
    # the selected zone is wholly disconnected, retain its swaths for UI
    # diagnostics but let _stitch_path return an empty path; this preserves the
    # established "selected zone exists but cannot be reached" contract and
    # never fabricates a straight connector through a wall.
    if not reachable.any():
        coverage_start_cell = _snap_to_free(
            coverage_free,
            _world_to_cell(robot_world, resolution, origin_x, origin_y),
            max_radius=max(width, height),
        )
        if coverage_start_cell is None:
            return CoveragePlan([], [], coverage_free, robot_world)
        reachable = largest_component(coverage_free, seed=coverage_start_cell)

    snapped_start = _cell_to_world(travel_start_cell, resolution, origin_x, origin_y)

    # #1 自动扫描方向: 沿可达区域的长边扫 → 行数最少 → 掉头最少 → 又快又稳.
    # axis="x" 行数∝y跨度, axis="y" 行数∝x跨度; 选垂直跨度更小的那个.
    auto_requested = (axis == "auto")   # 任意角主方向(下方)只在 auto 时启用
    forced_theta: Optional[float] = None
    if forced_angle_rad is not None:
        # [2012夜 补漏统一方向] 归一到 [-pi/2, pi/2)
        forced_theta = float(forced_angle_rad)
        while forced_theta >= math.pi / 2:
            forced_theta -= math.pi
        while forced_theta < -math.pi / 2:
            forced_theta += math.pi
    if axis == "auto":
        ys, xs = np.nonzero(reachable)
        if xs.size and ys.size:
            x_extent = int(xs.max() - xs.min())
            y_extent = int(ys.max() - ys.min())
            axis = "x" if x_extent >= y_extent else "y"
        else:
            axis = "x"
    if forced_theta is not None:
        # forced 近轴(≤8°走下面轴向路径)时, 轴向按 forced 角选; >8° 由旋转帧分支接管
        axis = "x" if abs(forced_theta) <= math.pi / 4 else "y"

    # Headland 内缩: 弓字 swath 在内缩区(整体离墙 ≥ 一个扫带)生成, 让弓字全程能被 MPPI
    # 跟住(不再贴 inflation 代价区被推偏); 最外圈边带留给下面的沿边环收尾。
    # 窄区(内缩后无 swath)兜底: 回退整块 reachable 生成 swath(等于旧行为), 绝不丢覆盖。
    # [2101用户反馈"填充范围少"→codexR9修正] headland 由整带(0.525)减到 max(半车宽0.34,
    # 0.4带): 外圈车道离墙 0.55+0.34=0.89m —— 车身外缘(半宽0.34)贴 inflation(0.55) 边缘,
    # 对 MPPI 外接圆碰撞半径(0.65)留 0.24m 跟踪余量(0.4带=0.21 时只剩 0.11, 会被代价压死,
    # codexR9-P1)。仍比原整带(离墙1.07)外扩 0.18m/侧, 边带留给②沿边(0.87)覆盖。
    headland_cells = max(1, int(round(max(0.34, swath_spacing_m * 0.4) / resolution)))
    interior = reachable & ~dilate_binary(~reachable, headland_cells)
    swath_mask = interior if bool(interior.any()) else reachable

    # [任意角主方向 2026-07-02] 区域主方向(PCA)偏离栅格 x/y 轴 >8° 时, 在"主方向对齐x轴"的
    # 旋转帧内生成扫带+起始带切分+弓字排序(直线沿长轴 → 车道数/掉头最少; 治斜放区域被
    # x/y 轴扫描切碎), 端点转回世界系后逐带在【原始mask】上校验裁剪(兜住旋转重采样毛边)。
    # 任何一步失败/异常 → ordered_swaths=None 回退原 x/y 逻辑, 绝不挡覆盖。
    ordered_swaths: Optional[List[Swath]] = None
    _bcd_ok = False   # [半圆掉头] BCD产出时掉头连接用半圆弧
    if no_rotate:
        rect_cells = False   # [2026-07-07] no_rotate 与 BCD 互斥: 强制轴对齐整块弓字
    # [2026-07-07 BCD矩形分割] 用户拍板"先分最大矩形块, 每块横平竖直依次扫":
    # 治PCA斜车道(L形区选了斜15°轴)在楔形角把两车道挤到6cm的发卡锐角→MPPI冻结24s。
    # 列扫描牛耕分区拆单元 → 每单元沿自身长轴的轴对齐弓字 → 单元间从机器人最近邻串接。
    # 任一步失败/异常 → 回落下方 PCA/轴向原逻辑, 绝不挡覆盖。
    if rect_cells and forced_theta is None:   # [审查F3] 补漏统一方向(forced)不走BCD
        try:
            jump = max(4, int(round(2.0 * swath_spacing_m / resolution)))
            cell_masks = _bcd_cells(swath_mask, jump)
            # [2026-07-07 同轴合并] 分单元的意义只在"不同车道朝向的区域各自成组"; 同朝向
            # 单元合并成一张mask再统一生成——墙面凸起切出的窄条/L拐角同轴块若各自成单元,
            # 产生"倒退连接段/单元间超长A*直角"(实测倒退2.17m=第一转角锐角三角)。合并后
            # 每列run天然被凸起裁短, 从左到右一遍扫完, 无单元间倒退。
            _groups: Dict[str, np.ndarray] = {}
            for cm in cell_masks:
                ys_c, xs_c = np.nonzero(cm)
                if xs_c.size == 0:
                    continue
                ax_c = "y" if (int(ys_c.max()) - int(ys_c.min())) >= (int(xs_c.max()) - int(xs_c.min())) else "x"
                if ax_c in _groups:
                    _groups[ax_c] = _groups[ax_c] | cm
                else:
                    _groups[ax_c] = cm.copy()
            # [codex审B2] 同轴合并必须按【连通分量】拆组: 同轴但不连通的两块(哑铃形
            # 两房一门)揉成一组, 弓字每一行都会左块→穿门→右块来回横跳。逐分量成组。
            _flat_groups = []
            for ax_c, gm in _groups.items():
                _rest = gm.copy()
                _n_comp = 0
                while _rest.any() and _n_comp < 16:
                    _ys0, _xs0 = np.nonzero(_rest)
                    _comp = largest_component(
                        _rest, seed=(int(_ys0[0]), int(_xs0[0])))
                    _flat_groups.append((ax_c, _comp))
                    _rest = _rest & ~_comp
                    _n_comp += 1
                if _rest.any():
                    # [codex终审] 余量并入最后一组: 超16分量绝不静默丢(漏扫), 宁可
                    # 排序次优也保覆盖完整。
                    _flat_groups.append((ax_c, _rest))
            # [2026-07-07 投影重叠合并] 修"分量拆分致同车道两段retrace 180°cusp":
            # L形被中间收窄掐成上下两块但共享同一批车道(同x竖直), 连通分量拆分把它们
            # 分开→排序上段扫完折回接下段=cusp(实测车在区外顶部起步时x3.74/x4.30两处180°)。
            # 按【车道垂直轴投影】(竖直车道=x, 水平车道=y)重叠者合并=同车道连续单扫;
            # 投影不重叠(哑铃两房并排)保持分开=不横跳穿门。两个需求都满足。
            _mg = []
            for _axc in set(_g[0] for _g in _flat_groups):
                _rs = []
                for _g in _flat_groups:
                    if _g[0] != _axc:
                        continue
                    _cy, _cx = np.nonzero(_g[1])
                    if _cx.size == 0:
                        continue
                    _pr = (int(_cx.min()), int(_cx.max())) if _axc == "y" else (int(_cy.min()), int(_cy.max()))
                    _rs.append([_pr[0], _pr[1], _g[1]])
                _usd = [False] * len(_rs)
                for _i in range(len(_rs)):
                    if _usd[_i]:
                        continue
                    _lo, _hi, _am = _rs[_i][0], _rs[_i][1], _rs[_i][2].copy()
                    _usd[_i] = True
                    _chg = True
                    while _chg:
                        _chg = False
                        for _j in range(len(_rs)):
                            if _usd[_j]:
                                continue
                            # [2026-07-07 大幅重叠才合] 只相切(L形左臂|右上块 x仅边界重叠)
                            # 不合→分两矩形各自扫(治发卡); 完全重叠(同车道被收窄掐断)才合→
                            # 治retrace。判据: 重叠格数 >= 较窄一方宽的一半。
                            _ov = min(_hi, _rs[_j][1]) - max(_lo, _rs[_j][0])
                            _wmin = min(_hi - _lo, _rs[_j][1] - _rs[_j][0])
                            if _ov >= 0.5 * max(1, _wmin):
                                _am = _am | _rs[_j][2]
                                _lo = min(_lo, _rs[_j][0])
                                _hi = max(_hi, _rs[_j][1])
                                _usd[_j] = True
                                _chg = True
                    _mg.append((_axc, _am))
            _flat_groups = _mg
            cell_info = []
            for ax_c, gm in _flat_groups:
                ys_c, xs_c = np.nonzero(gm)
                # [审查F5] 按组bbox裁剪生成: 扫描线相位锚到组自身
                r0c, r1c = int(ys_c.min()), int(ys_c.max())
                c0c, c1c = int(xs_c.min()), int(xs_c.max())
                # [2026-07-07 口袋救回] 碎带门槛1.8m会把墙凸起上方的口袋短带(实测左上
                # 角1.3m)剔掉——试验模式无补漏时=永久漏扫。BCD轴对齐下PCA碎带成因已除,
                # 门槛压到0.9只剔真碎屑(<0.9的由刷宽+沿边环兜)。
                sw = _generate_swaths_from_mask(
                    gm[r0c:r1c + 1, c0c:c1c + 1], resolution,
                    origin_x + c0c * resolution, origin_y + r0c * resolution,
                    swath_spacing_m=swath_spacing_m, axis=ax_c,
                    min_swath_m=min(min_swath_m, 0.5))   # [2026-07-07] 0.9->0.5: 右上块变窄处短车道(实测x5.98)survive
                if sw:
                    cx_c = origin_x + (float(xs_c.mean()) + 0.5) * resolution
                    cy_c = origin_y + (float(ys_c.mean()) + 0.5) * resolution
                    cell_info.append((sw, ax_c, (cx_c, cy_c), gm))
            acc: List[Swath] = []
            cur = robot_world
            first_cell = True
            # [2026-07-07 依次序] 用户拍板"从左到右依次": 单元按【扫掠轴坐标】排序依次扫
            # (主导车道竖直→按质心x排, 水平→按y排), 方向取机器人更近的那一端起步——
            # 质心最近邻串接实测产生"跳单元+长对角连接段横穿未扫车道+末尾折返"。
            _axes = [ci[1] for ci in cell_info]
            _dom_y = sum(1 for a in _axes if a == "y") * 2 >= max(1, len(_axes))
            _ki = 0 if _dom_y else 1
            if cell_info:
                _lo = min(ci[2][_ki] for ci in cell_info)
                _hi = max(ci[2][_ki] for ci in cell_info)
                _asc = abs(robot_world[_ki] - _lo) <= abs(robot_world[_ki] - _hi)
                cell_info.sort(key=lambda ci: ci[2][_ki], reverse=not _asc)
            for sw, ax_c, _c, _cm in cell_info:
                if first_cell:
                    # [2026-07-07 两阶段起步] 车在填充区【外】→ 不做车位切带(切带会造成
                    # 区中间入场+斜引入): 连接段(A*)先开到最近车道角落, 从角落起从一头
                    # 扫到另一头。车在区【内】才保留 yaw 感知的车位切带起步。
                    if _cell_is_free(swath_mask, _world_to_cell(
                            cur, resolution, origin_x, origin_y)):
                        sp_c, rest_c = _split_start_swath(
                            sw, cur, min_part_m=min_swath_m * 0.5,
                            start_heading=robot_yaw)
                    else:
                        sp_c, rest_c = [], sw
                    if sp_c:
                        pre_c = sp_c[:1]
                        ordered_c = pre_c + _order_swaths_boustrophedon(
                            sp_c[1:] + rest_c, pre_c[-1][1], ax_c, swath_spacing_m)
                    else:
                        ordered_c = _order_swaths_boustrophedon(
                            sw, cur, ax_c, swath_spacing_m)
                    first_cell = False
                else:
                    ordered_c = _order_swaths_boustrophedon(
                        sw, cur, ax_c, swath_spacing_m)
                if ordered_c:
                    acc.extend(ordered_c)
                    cur = ordered_c[-1][1]
            if acc:
                # [2026-07-07 就近端进入] 不等长车道下盲目交替方向会产生跨半场的长对角
                # 连接段(实测3.9m/3.3m两条斜穿区中间)。保持车道从左到右顺序不变, 每条带
                # 从【离当前位置近的那头】进入——等长车道时结果=原交替(半圆掉头照旧),
                # 不等长(墙凸起裁短/L拐角)时连接段最短。首带保留yaw选向不动。
                # [2026-07-07 出口前瞻] 只看进入端会对同车道拆段选错朝向→retrace 180°cusp
                # (实测x3.74上段从下端进扫上去, 出口远只能折回)。改: 朝向代价=进入距+出口
                # 到下一条带最近端距, 退化时自动选不折回的朝向。既短连接段又不retrace。
                _cur_e = acc[0][1]
                for _k in range(1, len(acc)):
                    _aa, _bb = acc[_k]
                    _nxt = acc[_k + 1] if _k + 1 < len(acc) else None

                    def _exit_cost(_ex):
                        if _nxt is None:
                            return 0.0
                        return min(_dist(_ex, _nxt[0]), _dist(_ex, _nxt[1]))

                    _cost_ab = _dist(_cur_e, _aa) + _exit_cost(_bb)   # aa→bb
                    _cost_ba = _dist(_cur_e, _bb) + _exit_cost(_aa)   # bb→aa
                    if _cost_ba < _cost_ab:
                        acc[_k] = (_bb, _aa)
                    _cur_e = acc[_k][1]
                # [2026-07-07 切向入场] 首带起点沿车道方向后延0.30m(进headland带):
                # 连接段瞄准延长点, 入车道成切向直线——治"A*连接段冲过车道口再钩回"
                # 的Z形锐角三角(实测第一转角)。延长点/沿线不在reachable内则不延。
                _a0, _b0 = acc[0]
                _dab = _dist(_a0, _b0)
                if _dab > 1e-6:
                    _ux2, _uy2 = (_b0[0] - _a0[0]) / _dab, (_b0[1] - _a0[1]) / _dab
                    _aext = (_a0[0] - _ux2 * 0.30, _a0[1] - _uy2 * 0.30)
                    # [codex审S6·二次] 整段校验(_aext→_a0 含端点)且双mask: 采样3点会漏
                    # 中间窄缝/_a0本身, 首连接成功后 body 仍可能穿出 travel 域。
                    _ok_ext = (
                        _world_segment_is_free(
                            reachable, _aext, _a0, resolution, origin_x, origin_y)
                        and _world_segment_is_free(
                            travel_reachable, _aext, _a0,
                            resolution, origin_x, origin_y))
                    if _ok_ext:
                        acc[0] = (_aext, _b0)
                ordered_swaths = acc
                _bcd_ok = True
        except Exception:  # noqa: BLE001 — 分割失败绝不挡覆盖, 回落原逻辑
            ordered_swaths = None
    if (ordered_swaths is None and (auto_requested or forced_theta is not None)
            and not no_rotate):   # [2026-07-07正解] no_rotate 禁PCA旋转分支, 保横平竖直
        try:
            theta = forced_theta if forced_theta is not None else _principal_angle(swath_mask)
            dev = min(abs(theta), abs(math.pi / 2 - abs(theta)))
            if dev > math.radians(8.0):
                rot = _make_rot_frame(swath_mask, resolution, origin_x, origin_y, theta)
                if rot is not None:
                    rmask, rox, roy, rcx, rcy, rct, rst = rot
                    raw_r = _generate_swaths_from_mask(
                        rmask, resolution, rox, roy,
                        swath_spacing_m=swath_spacing_m, axis="x",
                        min_swath_m=min_swath_m)
                    if raw_r:
                        def _to_world(q: Point) -> Point:
                            return (rcx + rct * q[0] - rst * q[1],
                                    rcy + rst * q[0] + rct * q[1])

                        def _to_rot(p: Point) -> Point:
                            return (rct * (p[0] - rcx) + rst * (p[1] - rcy),
                                    -rst * (p[0] - rcx) + rct * (p[1] - rcy))

                        robot_q = _to_rot(robot_world)
                        yaw_q = (robot_yaw - theta) if robot_yaw is not None else None
                        sp_r, rest_r = _split_start_swath(
                            raw_r, robot_q,
                            min_part_m=min_swath_m * 0.5,
                            start_heading=yaw_q)
                        if sp_r:
                            prefix_r = sp_r[:1]
                            ordered_r = prefix_r + _order_swaths_boustrophedon(
                                sp_r[1:] + rest_r, prefix_r[-1][1], "x", swath_spacing_m)
                        else:
                            ordered_r = _order_swaths_boustrophedon(
                                rest_r, robot_q, "x", swath_spacing_m)
                        clipped: List[Swath] = []
                        for _a, _b in ordered_r:
                            # 裁剪门限=生成门限(codex): 裁短到<min_swath 的算碎带, 丢给补漏,
                            # 否则 1.08-1.8m 碎带从旋转分支溜回来, F3 剔碎带白做
                            seg = _clip_swath_to_mask(
                                swath_mask, _to_world(_a), _to_world(_b),
                                resolution, origin_x, origin_y,
                                max(min_swath_m, 0.3))
                            if seg is not None:
                                clipped.append(seg)
                        if clipped:
                            ordered_swaths = clipped
            if (ordered_swaths is None and forced_theta is not None
                    and dev > math.radians(8.0)):
                # [0703审查#6-P0] 必须限定 dev>8°: 近轴 forced 本该走下面的轴向弓字,
                # 无门时兜底抢跑会把整块区域坍缩成过质心的【一条线】。
                # [0703 B1] 小碎块旋转重采样常产不出带 → 原来系统性回退【轴向】破坏统一
                # 方向(实测平行率1/4)。兜底: 沿 θ 过质心取最长自由段作"单笔画", 小块也
                # 保持主方向一致; 仍产不出才落轴向老路。
                ys0, xs0 = np.nonzero(swath_mask)
                if xs0.size:
                    cx0 = origin_x + (float(xs0.mean()) + 0.5) * resolution
                    cy0 = origin_y + (float(ys0.mean()) + 0.5) * resolution
                    ct0, st0 = math.cos(theta), math.sin(theta)
                    L0 = (max(int(xs0.max() - xs0.min()),
                              int(ys0.max() - ys0.min())) + 4) * resolution
                    seg0 = _clip_swath_to_mask(
                        swath_mask, (cx0 - ct0 * L0, cy0 - st0 * L0),
                        (cx0 + ct0 * L0, cy0 + st0 * L0),
                        resolution, origin_x, origin_y, max(min_swath_m, 0.3))
                    if seg0 is not None:
                        ordered_swaths = [seg0]
        except Exception:  # noqa: BLE001 — 任意角失败绝不挡覆盖, 回退轴向弓字
            ordered_swaths = None

    if ordered_swaths is None:
        raw_swaths = _generate_swaths_from_mask(
            swath_mask, resolution, origin_x, origin_y,
            swath_spacing_m=swath_spacing_m,
            axis=axis,
            min_swath_m=min_swath_m,
        )
        if not raw_swaths and swath_mask is not reachable:
            raw_swaths = _generate_swaths_from_mask(
                reachable, resolution, origin_x, origin_y,
                swath_spacing_m=swath_spacing_m, axis=axis, min_swath_m=min_swath_m)
        if not raw_swaths:
            return CoveragePlan([snapped_start], [], reachable, snapped_start)

        start_parts, rest_swaths = _split_start_swath(
            raw_swaths, robot_world,
            min_part_m=min_swath_m * 0.5,
            start_heading=robot_yaw,
        )
        if start_parts:
            # Keep the yaw-preferred starting half at the very front so the robot
            # sets off roughly along its current heading; boustrophedon-order the
            # rest (including the other half) from where that half ends. Without
            # this, _order_swaths_boustrophedon re-sorts from scratch and discards
            # the yaw-aware split, making the robot start ~150° off and spin.
            prefix = start_parts[:1]
            others = start_parts[1:] + rest_swaths
            _es = _order_swaths_extent_slab(
                others, prefix[-1][1], axis, swath_spacing_m, clip_polygon)
            ordered_swaths = prefix + (_es if _es is not None
                else _order_swaths_boustrophedon(
                    others, prefix[-1][1], axis, swath_spacing_m))
        else:
            _es = _order_swaths_extent_slab(
                rest_swaths, robot_world, axis, swath_spacing_m, clip_polygon)
            ordered_swaths = (_es if _es is not None
                else _order_swaths_boustrophedon(
                    rest_swaths, robot_world, axis, swath_spacing_m))
    raw_path = _stitch_path(
        free_mask=reachable,
        entry_free_mask=travel_reachable,
        swaths=ordered_swaths,
        start=robot_world if _cell_is_free(travel_reachable, _world_to_cell(
            robot_world, resolution, origin_x, origin_y)) else snapped_start,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        path_step_m=path_step_m,
        semicircle_uturns=_bcd_ok,
    )
    if _bcd_ok and len(raw_path) >= 3:
        # [2026-07-07 全圆角v2] BCD路径【跳过】_smooth_path_corners——实测它会把"上去又
        # 回来"的发卡形短车道整段吃掉(墙凸起裁短的车道0在smooth后从路径消失)。BCD的
        # 平滑由 RDP(直道坍到端点、弧保留)+圆角级联(>25°拐角贝塞尔弧, mask校验过不了
        # 保留)+按path_step加密 独立完成; 任一步异常回退smooth老管线。
        path = raw_path
        try:
            # 圆角校验mask放宽(同沿边B升级款): dilate1格∩底线(留边-1格+clip+禁区照剔)——
            # 斜边端头的弧只差几cm被严格reachable打回, 放宽后能圆; 硬边界仍绝不越。
            _relax_f = dilate_binary(reachable, 1)
            _relax_f &= build_coverage_free_mask(
                data=data, width=width, height=height, resolution=resolution,
                robot_radius_m=max(0.0, robot_radius_m - resolution),
                clip_polygon=clip_polygon, blocked_polygons=blocked_polygons,
                origin_x=origin_x, origin_y=origin_y)
            # [codex审W5] RDP用放宽mask+eps0.05: A*八连通楼梯(0.08格)偏差≈0.03-0.06且弦贴
            # 边界被严格mask打回 → 0.03/严格下楼梯原样保留, 连接段仍小锯齿。
            _simp = _rdp_masked(raw_path, 0.05, _relax_f, resolution, origin_x, origin_y)
            # [2026-07-07 去回折小尖] BCD跳过_smooth_path_corners时把它内含的_remove_reversal_spurs
            # 也丢了→残留极小out-and-back尖(拆段/短pocket/连接段过冲, 实测170-180°cusp随起步位置
            # 变)。加回: max_shortcut0.60只删≤0.6m的回折尖(车道>0.9m不吃; 半圆是渐变多点非单点
            # 尖不受影响)。放rdp后fillet前。
            _simp = _remove_reversal_spurs(
                _simp, _relax_f, resolution, origin_x, origin_y,
                max_shortcut_m=0.60)
            _fil = _fillet_corners(_simp, _relax_f, resolution, origin_x, origin_y,
                                   radius_m=0.25, path_step_m=path_step_m)
            path = _densify_polyline(_fil, path_step_m)
            # [2026-07-07 稠密路径终清尖] 尖多来自A*连接段(rdp前spur跑在稀疏点抓不住)——
            # 在最终稠密路径上再删一遍窄回折尖(shortcut0.35只删≤0.35m尖; 半圆/S弯是渐变
            # 每步~11°<95°阈值不受影响)+重新加密。迭代到无尖。
            for _sp_it in range(3):
                _n0 = len(path)
                path = _remove_reversal_spurs(
                    path, _relax_f, resolution, origin_x, origin_y,
                    max_shortcut_m=0.35)
                path = _densify_polyline(path, path_step_m)
                if len(path) >= _n0:
                    break
        except Exception:  # noqa: BLE001 — 圆角失败回退smooth老管线, 绝不挡覆盖
            path = _smooth_path_corners(
                raw_path, reachable, resolution, origin_x, origin_y,
                turn_radius_m=turn_radius_m, path_step_m=path_step_m,
            )
    else:
        path = _smooth_path_corners(
            raw_path, reachable, resolution, origin_x, origin_y,
            turn_radius_m=turn_radius_m, path_step_m=path_step_m,
        )
    return CoveragePlan(path, ordered_swaths, reachable, snapped_start)


def plan_perimeter_coverage(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_world: Point,
    obstacle_clearance_m: float,
    zone_inset_m: float = 0.0,
    clip_polygon: Optional[Sequence[Point]] = None,
    blocked_polygons: Optional[Sequence[Sequence[Point]]] = None,
    path_step_m: float = 0.10,
    min_ring_cells: int = 8,
) -> List[Point]:
    """沿边收尾路径(独立于内部弓字, 作为覆盖任务【所有内部扫完后】的最终收尾阶段)。

    全程走 Nav2/MPPI, 故路径必须避开 inflation 代价带否则 MPPI 被高代价压死卡住(实测 9.4s):
    - obstacle_clearance_m: 沿边路径离【真实障碍/墙】的车心留边(传 nav2 inflation_radius + 安全
      裕量, 如 0.75m; inflation 是代价梯度非硬边界, 这里留够远让 MPPI 不贴障碍) → 近障处自动内收。
    - zone_inset_m: 对【清扫区边线 clip_polygon】的内偏(≈半车宽, 如 0.34m) → 开阔处车身外侧压线。
    两约束取交=取更内收者。从 robot_world 就近进入, 沿外边界+障碍岛洞各绕一圈, 环/弧间 A* 连接。
    返回车心 polyline(从 robot_world 出发); 无可行环返回 []。
    """
    peri_mask = build_coverage_free_mask(
        data=data, width=width, height=height, resolution=resolution,
        robot_radius_m=obstacle_clearance_m,
        clip_polygon=None, blocked_polygons=blocked_polygons,
        origin_x=origin_x, origin_y=origin_y)
    if clip_polygon is not None:
        zone = polygon_to_mask(clip_polygon, width, height, resolution, origin_x, origin_y)
        inset_cells = max(0, int(round(zone_inset_m / resolution)))
        if inset_cells > 0:
            zone = zone & ~dilate_binary(~zone, inset_cells)
        peri_mask = peri_mask & zone
    if not peri_mask.any():
        return []
    # 只保留机器人能进入的那个连通块(seed 取离 robot_world 最近的 peri_mask cell)
    seed = _snap_to_free(
        peri_mask, _world_to_cell(robot_world, resolution, origin_x, origin_y),
        max_radius=max(width, height))
    if seed is None:
        return []
    peri_mask = largest_component(peri_mask, seed=seed)
    rings = _trace_perimeter_rings(
        peri_mask, resolution, origin_x, origin_y,
        step_m=path_step_m, min_ring_cells=min_ring_cells)
    # [gap#1修复 2026-07-02] 连接段起点吸附: 车若正站在 0.55~0.87m 留边带内(peri_mask 之外),
    # 原来 _assemble 从车位起连、_polyline_is_free 必失败 → 所有环连不上 → 沿边被静默跳过。
    # 现从 mask 内离车最近的自由点起步(车→起步点那段由发送侧"当前车位接入"交给 Nav2 开过去)。
    start_pt = robot_world
    if not _cell_is_free(peri_mask, _world_to_cell(robot_world, resolution, origin_x, origin_y)):
        start_pt = _cell_to_world(seed, resolution, origin_x, origin_y)
    # 环/弧间连接段也走 peri_mask(clearance 区, 不贴障碍); peri_mask 已连通, A* 能在其内连各段。
    ring_path = _assemble_perimeter_path(
        start_pt, rings, peri_mask, resolution, origin_x, origin_y, path_step_m)
    # [0703晨 平滑 → 2026-07-07 B升级] 环是栅格描迹: 直墙段0.05格锯齿(实测3.5↔3.58横跳
    # 8cm, 31个直角)。旧RDP(eps0.06, 严格mask)治不了: 锯齿幅0.08>eps, 且拉直弦贴锯齿墙侧
    # 被mask校验打回。修: 弦校验mask放宽1格 且∩底线mask(留边=clearance-1格): 两mask同用
    # 4连通菱形核 → 相对基线【任何方向含45°对角】最坏只退1格0.08m, 落0.80历史兜底档之上;
    # 且底线mask带原clip+禁区照剔——peri_mask边界里 禁区/未知区/清扫区内偏线 三类零缓冲,
    # 单纯dilate会越进去[审查F1]。+ eps0.12收锯齿 + 拐角圆弧(过不了保留直角) + 按 path_step
    # 重新加密(执行侧吸附/堵点检测/进度统计依赖密集点——只能"直而密"不能"疏")。
    if len(ring_path) >= 3:
        relax = dilate_binary(peri_mask, 1)
        floor_mask = build_coverage_free_mask(
            data=data, width=width, height=height, resolution=resolution,
            robot_radius_m=max(0.0, obstacle_clearance_m - resolution),
            clip_polygon=clip_polygon, blocked_polygons=blocked_polygons,
            origin_x=origin_x, origin_y=origin_y)
        relax &= floor_mask
        simp = _rdp_masked(ring_path, 0.12, relax, resolution, origin_x, origin_y)
        simp = _fillet_corners(simp, relax, resolution, origin_x, origin_y,
                               radius_m=0.25, path_step_m=path_step_m)
        ring_path = _densify_polyline(simp, path_step_m)
    return ring_path


def paint_lethal_into_map(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    lc_data,
    lc_width: int,
    lc_height: int,
    lc_resolution: float,
    lc_origin_x: float,
    lc_origin_y: float,
    lethal_min: int = 100,
    transform_to_map: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[bytes, int]:
    """[实时避障并入规划 2026-07-02 二期#1] 把 local costmap 的 lethal 格(实时障碍,
    含建图后新出现的物体)涂进静态占用图【副本】, 返回 (新data, 涂格数)。
    涂完走原有 build_coverage_free_mask 腐蚀 → 覆盖路径/连接段/沿边/补漏天然绕开实时障碍。
    只涂 lethal(>=100), 不涂 inflation(<99)/inscribed(99) —— 留障由腐蚀统一加, 避免双重膨胀。
    治0702实测: 区域内新物体规划看不见 → 路径反复往上带 → MPPI连环Optimizer fail烧光恢复预算。"""
    # 兼容 bytes(测试)与 rclpy array('b')(实机 OccupancyGrid.data), 均按 int8 语义解析(-1=unknown)
    try:
        lc = np.frombuffer(bytes(lc_data), dtype=np.int8)
    except (TypeError, ValueError):
        lc = np.asarray(lc_data, dtype=np.int8)
    lc = lc.reshape(lc_height, lc_width)
    lys, lxs = np.nonzero(lc >= lethal_min)
    if lxs.size == 0:
        return data, 0
    wx = lc_origin_x + (lxs.astype(np.float64) + 0.5) * lc_resolution
    wy = lc_origin_y + (lys.astype(np.float64) + 0.5) * lc_resolution
    # [codex P0] local costmap 帧(odom)→静态图帧(map) 2D变换: p_m = t + R·p_o。
    # 不传(=None)按同帧处理(单位变换)。
    if transform_to_map is not None:
        tx, ty, ct, st = transform_to_map
        wx, wy = tx + ct * wx - st * wy, ty + st * wx + ct * wy
    col = np.floor((wx - origin_x) / resolution).astype(np.int64)
    row = np.floor((wy - origin_y) / resolution).astype(np.int64)
    ok = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    if not bool(ok.any()):
        return data, 0
    arr = np.frombuffer(bytes(data), dtype=np.int8).reshape(height, width).copy()
    arr[row[ok], col[ok]] = 100
    return arr.tobytes(), int(ok.sum())


def _bcd_cells(mask: np.ndarray, jump_cells: int,
               min_cell_area: int = 12) -> List[np.ndarray]:
    """[2026-07-07 BCD矩形分割] 列扫描把 mask 分解为若干"矩形样"单元(牛耕分区)。
    每列取自由 run 区间, 与上一列活动单元 1-1 匹配且端点跳变≤jump_cells 视为延续;
    分叉/合并/端点大跳 → 关闭旧单元开新单元。斜边(端点逐列渐变)不切分;
    L形拐角(端点一次跳几十格)切分——正是想要的分割点。返回各单元 bool mask。"""
    h, w = mask.shape
    cells: List[np.ndarray] = []
    active: List[Tuple[int, int, np.ndarray]] = []   # (lo, hi, cell_mask)
    for j in range(w):
        runs = list(_true_runs(mask[:, j], 1))
        run_hits = [[] for _ in runs]
        act_hits = [[] for _ in active]
        for ri, (lo, hi) in enumerate(runs):
            for ai, (alo, ahi, _cm) in enumerate(active):
                if lo <= ahi and alo <= hi:   # 区间重叠
                    run_hits[ri].append(ai)
                    act_hits[ai].append(ri)
        new_active: List[Tuple[int, int, np.ndarray]] = []
        continued = [False] * len(active)
        used_run = [False] * len(runs)
        for ri, (lo, hi) in enumerate(runs):
            ais = run_hits[ri]
            if (len(ais) == 1 and len(act_hits[ais[0]]) == 1
                    and abs(lo - active[ais[0]][0]) <= jump_cells
                    and abs(hi - active[ais[0]][1]) <= jump_cells):
                _alo, _ahi, cm = active[ais[0]]
                cm[lo:hi + 1, j] = True
                new_active.append((lo, hi, cm))
                continued[ais[0]] = True
                used_run[ri] = True
        for ai, (_alo, _ahi, cm) in enumerate(active):
            if not continued[ai] and int(cm.sum()) >= min_cell_area:
                cells.append(cm)
        for ri, (lo, hi) in enumerate(runs):
            if not used_run[ri]:
                cm = np.zeros_like(mask)
                cm[lo:hi + 1, j] = True
                new_active.append((lo, hi, cm))
        active = new_active
    for (_alo, _ahi, cm) in active:
        if int(cm.sum()) >= min_cell_area:
            cells.append(cm)
    return cells


def _fillet_corners(
    pts: List[Point],
    mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    radius_m: float,
    path_step_m: float,
    min_turn_deg: float = 25.0,
) -> List[Point]:
    """[2026-07-07 沿边平滑] 拐角圆弧化: 转角>min_turn_deg 的顶点两侧各截 r(受邻段长
    45%上限保护), 二次贝塞尔过渡; 弧上采样点全过 mask 校验, 不过则保留原直角。"""
    if len(pts) < 3:
        return list(pts)
    out: List[Point] = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0 = out[-1]
        p1 = pts[i]
        p2 = pts[i + 1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1 = math.hypot(v1[0], v1[1])
        l2 = math.hypot(v2[0], v2[1])
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        if turn < min_turn_deg:
            out.append(p1)
            continue
        r0 = min(radius_m, 0.45 * l1, 0.45 * l2)
        if r0 < path_step_m * 0.5:   # [2026-07-07] 门槛0.10→0.05: 短段角也吃小圆角(45°+45°远好过90°)
            out.append(p1)
            continue
        done = False
        # [2026-07-07 级联] 大半径被mask挡(斜边端头)→逐级缩小重试: 小圆角也远好过硬直角
        for r in (r0, r0 * 0.75, r0 * 0.55, r0 * 0.4, r0 * 0.28):
            if r < 0.04:
                break
            a = (p1[0] - v1[0] / l1 * r, p1[1] - v1[1] / l1 * r)
            b = (p1[0] + v2[0] / l2 * r, p1[1] + v2[1] / l2 * r)
            # [审查F2] 采样步距0.04m: 0.10步距相邻点可跨过未检格
            n = max(8, int(math.ceil((2.0 * r) / 0.04)))
            arc: List[Point] = []
            ok = True
            for k in range(n + 1):
                t = k / n
                x = (1 - t) * (1 - t) * a[0] + 2 * (1 - t) * t * p1[0] + t * t * b[0]
                y = (1 - t) * (1 - t) * a[1] + 2 * (1 - t) * t * p1[1] + t * t * b[1]
                if not _point_is_free(mask, (x, y), resolution, origin_x, origin_y):
                    ok = False
                    break
                arc.append((x, y))
            if ok:
                out.extend(arc)
                done = True
                break
        if not done:
            out.append(p1)
    out.append(pts[-1])
    return out


def _principal_angle(mask: np.ndarray) -> float:
    """区域主方向角(弧度, [-pi/2, pi/2), 相对世界x轴)。PCA of free cells。
    栅格行列与世界xy同向等距, 故栅格系PCA角==世界系角。细长区域给出长轴方向;
    近方形区域角度不稳定, 由调用方的 ±8° 轴向豁免兜底(方形沿哪个方向扫都差不多)。"""
    ys, xs = np.nonzero(mask)
    if xs.size < 50:
        return 0.0
    x = xs.astype(np.float64) - float(xs.mean())
    y = ys.astype(np.float64) - float(ys.mean())
    cxx = float((x * x).mean())
    cyy = float((y * y).mean())
    cxy = float((x * y).mean())
    theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)   # 主特征向量方向
    while theta >= math.pi / 2:
        theta -= math.pi
    while theta < -math.pi / 2:
        theta += math.pi
    return theta


def _make_rot_frame(
    mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    theta: float,
):
    """构建"主方向对齐x轴"的旋转栅格帧 (2026-07-02 任意角弓字)。
    旋转世界坐标定义: q = R(-θ)·(p-c), p = c + R(θ)·q, c=区域质心(世界系)。
    返回 (rmask, r_origin_x, r_origin_y, cx, cy, cosθ, sinθ); 无自由格返回 None。
    rmask 用最近邻反向采样(旋转格中心→原世界→原格), 采样毛边由调用方对最终扫带
    在【原始mask】上逐段校验/裁剪兜底。"""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    wx = origin_x + (xs.astype(np.float64) + 0.5) * resolution
    wy = origin_y + (ys.astype(np.float64) + 0.5) * resolution
    cx = float(wx.mean())
    cy = float(wy.mean())
    ct, st = math.cos(theta), math.sin(theta)
    qx = ct * (wx - cx) + st * (wy - cy)
    qy = -st * (wx - cx) + ct * (wy - cy)
    pad = 2.0 * resolution
    qx0 = float(qx.min()) - pad
    qy0 = float(qy.min()) - pad
    rw = max(4, int(math.ceil((float(qx.max()) + pad - qx0) / resolution)))
    rh = max(4, int(math.ceil((float(qy.max()) + pad - qy0) / resolution)))
    if rw * rh > 16_000_000:   # 防超大图内存爆(4000x4000)
        return None
    jj, ii = np.meshgrid(np.arange(rw), np.arange(rh))
    qcx = qx0 + (jj + 0.5) * resolution
    qcy = qy0 + (ii + 0.5) * resolution
    pxw = cx + ct * qcx - st * qcy
    pyw = cy + st * qcx + ct * qcy
    col = np.floor((pxw - origin_x) / resolution).astype(np.int64)
    row = np.floor((pyw - origin_y) / resolution).astype(np.int64)
    valid = (row >= 0) & (row < mask.shape[0]) & (col >= 0) & (col < mask.shape[1])
    rmask = np.zeros((rh, rw), dtype=bool)
    rmask[valid] = mask[row[valid], col[valid]]
    return rmask, qx0, qy0, cx, cy, ct, st


def _clip_swath_to_mask(
    mask: np.ndarray,
    a: Point,
    b: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
    min_len_m: float,
) -> Optional[Swath]:
    """把(可能因旋转重采样毛边越界的)扫带按原始mask裁剪: 沿段采样, 取最长自由子段。
    子段短于 min_len_m 返回 None(丢弃)。保持 a→b 原方向。"""
    d = _dist(a, b)
    if d < 1e-6:
        return None
    n = max(2, int(math.ceil(d / (resolution * 0.5))) + 1)
    free_flags = []
    for i in range(n):
        t = i / (n - 1)
        p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        free_flags.append(_point_is_free(mask, p, resolution, origin_x, origin_y))
    best_i0, best_i1, cur0 = -1, -1, None
    for i, f in enumerate(free_flags + [False]):
        if f and cur0 is None:
            cur0 = i
        elif not f and cur0 is not None:
            if best_i0 < 0 or (i - 1 - cur0) > (best_i1 - best_i0):
                best_i0, best_i1 = cur0, i - 1
            cur0 = None
    if best_i0 < 0:
        return None
    t0, t1 = best_i0 / (n - 1), best_i1 / (n - 1)
    p0 = (a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]))
    p1 = (a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1]))
    if _dist(p0, p1) < min_len_m:
        return None
    return (p0, p1)


def _generate_swaths_from_mask(
    free: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    swath_spacing_m: float,
    axis: str,
    min_swath_m: float,
) -> List[Swath]:
    spacing_cells = max(1, int(round(swath_spacing_m / resolution)))
    min_run_cells = max(2, int(round(min_swath_m / resolution)))
    swaths: List[Swath] = []
    height, width = free.shape

    if axis not in ("x", "y"):
        axis = "x"

    # 末行/末列补扫: 扫描线从 spacing_cells//2 起步, 步进 spacing_cells, 最后一条
    # 落在 < (height|width) 的最大索引。若它到边界的距离 > 半个脚印(≈spacing_cells//2),
    # 边界那条带就漏扫(每隔 spacing_cells 周期性出现 1~3 格的缝)。补一条贴边扫描线,
    # 但只在确实留缝(gap > spacing_cells//2)且不与最后一条重合时加, 避免重复掉头。
    def _scan_indices(extent: int) -> List[int]:
        idxs = list(range(spacing_cells // 2, extent, spacing_cells))
        last_valid = extent - 1
        if not idxs:
            if last_valid >= 0:
                idxs.append(last_valid)
            return idxs
        gap = last_valid - idxs[-1]
        if gap > spacing_cells // 2 and idxs[-1] != last_valid:
            idxs.append(last_valid)
        return idxs

    if axis == "x":
        for row in _scan_indices(height):
            for lo, hi in _true_runs(free[row, :], min_run_cells):
                swaths.append((
                    _cell_to_world((row, lo), resolution, origin_x, origin_y),
                    _cell_to_world((row, hi), resolution, origin_x, origin_y),
                ))
    else:
        for col in _scan_indices(width):
            for lo, hi in _true_runs(free[:, col], min_run_cells):
                swaths.append((
                    _cell_to_world((lo, col), resolution, origin_x, origin_y),
                    _cell_to_world((hi, col), resolution, origin_x, origin_y),
                ))
    return swaths


def _true_runs(line: np.ndarray, min_len: int) -> Iterable[Tuple[int, int]]:
    in_run = False
    start = 0
    for i, val in enumerate(line):
        if bool(val) and not in_run:
            start = i
            in_run = True
        elif not bool(val) and in_run:
            if i - start >= min_len:
                yield start, i - 1
            in_run = False
    if in_run and len(line) - start >= min_len:
        yield start, len(line) - 1


def _order_swaths_nearest(swaths: Sequence[Swath], start: Point) -> List[Swath]:
    remaining = list(swaths)
    ordered: List[Swath] = []
    cur = start
    while remaining:
        best_i = 0
        best_reverse = False
        best_d = float("inf")
        for i, (a, b) in enumerate(remaining):
            da = _dist(cur, a)
            db = _dist(cur, b)
            if da < best_d:
                best_i = i
                best_reverse = False
                best_d = da
            if db < best_d:
                best_i = i
                best_reverse = True
                best_d = db
        a, b = remaining.pop(best_i)
        if best_reverse:
            a, b = b, a
        ordered.append((a, b))
        cur = b
    return ordered


# ---------------------------------------------------------------------------
# [2026-07-07 板块延伸带排序] 治L/凹形填充路径"俯冲发卡尖": 见文件顶注释与
# coverage-cleaning-test记忆。核心=按along跨度把车道分长(贯穿臂)/短(仅块内)两带,
# 长带成对在臂底做正常U掉头(0.56m间距可圆)、末条顶部出带与短带无缝续接; 错位离群
# 车道排成末端悬臂(spur); DP优化朝向+代价评分选最优候选。**安全护栏**: 只接受覆盖
# 全部车道的候选(len(seq)==len(swaths)), 否则/异常/无凹口 返回None → 调用方回退原弓字。
# 8-agent对抗工作流产出+主控真覆盖率复核选定(其余策略靠掉覆盖换零尖被淘汰)。
# ---------------------------------------------------------------------------
def reflex_levels(poly, along, min_sep=0.6):
    """裁剪多边形反射(凹)顶点在指定轴的投影坐标(along=1取y, 0取x)。L的凹口→[y]或[x]。"""
    n = len(poly)
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    ccw = area > 0
    lv = []
    for i in range(n):
        p0 = poly[(i - 1) % n]; p1 = poly[i]; p2 = poly[(i + 1) % n]
        ax, ay = p1[0] - p0[0], p1[1] - p0[1]
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        cross = ax * by - ay * bx
        if (cross < 0) if ccw else (cross > 0):
            lv.append(p1[along])
    lv = sorted(set(round(v, 2) for v in lv))
    out = []
    for v in lv:
        if out and abs(v - out[-1]) < min_sep:
            continue
        out.append(v)
    return out


def _order_swaths_extent_slab(swaths, start, axis, spacing, poly):
    try:
        along = 0 if axis == "x" else 1
        perp  = 1 - along
        if not swaths:
            return None
        tol = max(1e-6, float(spacing) * 0.5)
        spc = float(spacing)

        def perp_of(s): return 0.5 * (s[0][perp] + s[1][perp])
        def a_lo(s):    return min(s[0][along], s[1][along])
        def a_hi(s):    return max(s[0][along], s[1][along])

        items = sorted(swaths, key=perp_of)
        lanes = [[items[0]]]
        for s in items[1:]:
            if abs(perp_of(s) - perp_of(lanes[-1][-1])) <= tol:
                lanes[-1].append(s)
            else:
                lanes.append([s])

        rl = reflex_levels(poly, along)
        thresh = rl[0] if rl else None

        L = []
        for lane in lanes:
            lo = min(a_lo(s) for s in lane)
            hi = max(a_hi(s) for s in lane)
            L.append({'lane': lane, 'perp': perp_of(lane[0]), 'lo': lo, 'hi': hi})
        L.sort(key=lambda d: d['perp'])
        n = len(L)
        if n < 2 or thresh is None:
            return None

        def emit(li, up):
            fs = sorted(li['lane'], key=a_lo, reverse=not up)
            out = []
            for s in fs:
                a, b = s
                lo_first = (a[along] <= b[along])
                if up and not lo_first: a, b = b, a
                if (not up) and lo_first: a, b = b, a
                out.append((a, b))
            return out
        def entry(li, up): return li['lo'] if up else li['hi']
        def exit_(li, up): return li['hi'] if up else li['lo']

        long_ids  = [k for k in range(n) if L[k]['lo'] < thresh - 0.3]
        short_ids = [k for k in range(n) if L[k]['lo'] >= thresh - 0.3]
        if not long_ids or not short_ids:
            return None

        # Post-process a candidate: if the robot start lies in the interior of the FIRST
        # emitted swath (perp-aligned, along strictly inside), re-emit that swath as two
        # halves from the start point so the robot sets off toward the swath's far end
        # first and never backtracks past its start (kills the entry-overshoot cusp).
        def fix_start(seq):
            if not seq:
                return seq
            a, b = seq[0]
            if abs(a[perp]-start[perp]) > 0.20:
                return seq
            lo_a = min(a[along], b[along]); hi_a = max(a[along], b[along])
            sa = start[along]
            if not (lo_a + 0.12 < sa < hi_a - 0.12):
                return seq
            # split point on the lane axis at start's along-position, keeping lane perp
            pv = a[perp]
            if along == 1:
                sp_pt = (pv, sa)
            else:
                sp_pt = (sa, pv)
            # original direction: a -> b. We go from start toward b first (continue the
            # lane's intended forward direction), then cover start -> a.
            head = (sp_pt, b)
            tail = (sp_pt, a)
            newseq = [head, tail] + list(seq[1:])
            return newseq

        # ---- faithful proxy over an emitted sequence incl. start connector, plus the
        # first turn (start -> first entry -> first body) and the arm U-turn tightness ----
        def score(seq):
            cost = 0.0
            conns = [(start, seq[0][0], seq[0][1])]
            for i in range(len(seq)-1):
                conns.append((seq[i][1], seq[i+1][0], seq[i+1][1]))
            for j, (E, N, F) in enumerate(conns):
                lat = abs(E[perp]-N[perp]); same = lat < 1e-6
                dEN = E[along]-N[along]; dFN = F[along]-N[along]
                w = 1.6 if j == 0 else 1.0    # weight the very first turn a bit more
                if dEN*dFN > 1e-9:
                    if same: cost += w*(0.8 + min(abs(dEN)/0.15, 25.0))
                    else:    cost += w*(1.0 + min(abs(dEN)/max(lat,spc), 25.0))
                aj = abs(dEN)
                if not same and aj > 0.9 and (aj/max(lat,1e-3)) > 1.3:
                    cost += w*(0.7 + min(aj/max(lat,1e-3), 25.0))
                # tight U-turn: adjacent-lane turn where the two ends are vertically
                # offset relative to the lateral gap (turn radius shrinks). Penalize when
                # the connector's along-offset exceeds ~half the lateral gap.
                if not same and dEN*dFN <= 1e-9:  # a genuine U-turn (no overshoot)
                    off = abs(dEN)
                    if off > 0.45*lat + 0.05:
                        cost += 0.5 + min((off - 0.45*lat)/max(lat,1e-3), 10.0)
            jump = 0.0; prev = start
            for a,b in seq:
                jump += abs(prev[along]-a[along]) + abs(prev[perp]-a[perp]); prev=b
            return cost + 1e-4*jump

        # ---- build the canonical valid route: the long (arm-reaching) lanes meet each
        # other at the arm bottom and the last one exits at the TOP; the short (block)
        # lanes then boustrophedon, seamed at the top. Directions are optimized by DP;
        # misaligned outlier lanes are routed as terminal spurs. ----
        INF = float('inf')
        def rev_cost(lp, pup, ln, nup):
            ex = exit_(lp, pup); en = entry(ln, nup)
            far_n = exit_(ln, nup); body_p = entry(lp, pup); c = 0.0
            lat = abs(lp['perp'] - ln['perp']) + 1e-6
            dEN = ex-en; dFN = far_n-en
            if dEN*dFN > 1e-9: c += 1.0 + min(abs(dEN)/spc, 30.0)
            dNE = en-ex; dBE = body_p-ex
            if dNE*dBE > 1e-9: c += 1.0 + min(abs(dNE)/spc, 30.0)
            # tight U-turn: turn ends offset in 'along' by more than ~half the lateral gap
            off = abs(ex - en)
            if off > 0.45*lat + 0.05:
                c += 0.6 + min((off - 0.45*lat)/lat, 10.0)
            return c
        def dp_short(chain, seed_exit_along, seed_body_up):
            m = len(chain)
            if m == 0: return []
            cst=[[INF,INF] for _ in range(m)]; par=[[-1,-1] for _ in range(m)]
            # seed: previous "lane" is the seam exit at top; treat as a virtual lane whose
            # exit is at seam_along and body was upward (came up the long lane).
            for d in (0,1):
                en = entry(chain[0], d==1); far=exit_(chain[0], d==1)
                c=0.0
                dEN = seed_exit_along-en; dFN = far-en
                if dEN*dFN>1e-9: c+=1.0+min(abs(dEN)/spc,30.0)
                cst[0][d]=c
            for i in range(1,m):
                for d in (0,1):
                    for pd in (0,1):
                        if cst[i-1][pd]==INF: continue
                        c=cst[i-1][pd]+rev_cost(chain[i-1],pd==1,chain[i],d==1)
                        if c<cst[i][d]: cst[i][d]=c; par[i][d]=pd
            ed = 0 if cst[m-1][0]<=cst[m-1][1] else 1
            dirs=[0]*m; dirs[m-1]=ed
            for i in range(m-1,0,-1): dirs[i-1]=par[i][dirs[i]]
            return [sw for i in range(m) for sw in emit(chain[i], dirs[i]==1)]

        # short-lane visiting orders to try: perp-asc, and variants that push an
        # outlier (much-truncated, high-lo) short lane to the END so its misaligned
        # bottom becomes a terminal spur (no tight turn after it).
        def short_order_variants():
            base = list(short_ids)  # perp ascending
            variants = [base]
            if len(base) >= 3:
                # move the highest-lo short lane to the end
                hi_lo = max(base, key=lambda k: L[k]['lo'])
                if L[hi_lo]['lo'] - min(L[k]['lo'] for k in base) > 0.5:
                    v = [k for k in base if k != hi_lo] + [hi_lo]
                    variants.append(v)
                    # also its reverse-perp version
                    v2 = [k for k in base[::-1] if k != hi_lo] + [hi_lo]
                    variants.append(v2)
            variants.append(base[::-1])
            return variants

        # Build the long-lane prefix for a given long-lane visiting order, orienting so
        # the pair meets at the arm bottom and the LAST long lane exits at the top (seam).
        def long_prefix(order):
            m = len(order); flags = {}
            for i, k in enumerate(order):
                flags[k] = ((m-1-i) % 2 == 0)   # last -> up (exit top)
            seq = []
            for k in order:
                seq += emit(L[k], flags[k])
            seam = exit_(L[order[-1]], flags[order[-1]])
            return seq, seam

        # long-lane visiting-order variants (both perp directions). For 2 long lanes with
        # MISALIGNED arm bottoms, order matters for which one exits at the top.
        long_orders = [long_ids[:], long_ids[::-1]]

        fwd_variants = []
        for lo_order in long_orders:
            lseq, seam = long_prefix(lo_order)
            for so in short_order_variants():
                sc = [L[k] for k in so]
                fwd_variants.append(lseq + dp_short(sc, seam, True))

        # If a long lane's arm bottom is much higher than the deepest long lane, its arm
        # U-turn with the deep lane is tight. Route it as a TERMINAL spur (entered at the
        # top, exiting at its arm bottom = route end, no tight turn after).
        if len(long_ids) >= 2:
            min_lo = min(L[k]['lo'] for k in long_ids)
            outlier = max(long_ids, key=lambda k: L[k]['lo'])
            if L[outlier]['lo'] - min_lo > 0.45:
                core_long = [k for k in long_ids if k != outlier]
                lseq, seam = long_prefix(core_long)   # deep long lane(s), exit top
                for so in short_order_variants():
                    sc = [L[k] for k in so]
                    body = lseq + dp_short(sc, seam, True)
                    # append the outlier long lane as a terminal spur: enter TOP, exit arm
                    spur = emit(L[outlier], False)   # down: enter hi(top), exit lo(arm)
                    fwd_variants.append(body + spur)

        fwd = fwd_variants[0]

        # candidates: every forward variant and its full reverse
        candidates = []
        for fv in fwd_variants:
            candidates.append(fv)
            candidates.append([(b, a) for (a, b) in reversed(fv)])

        # Greedy fallback candidates (perp asc / desc) with nearest-end orientation, in
        # case truncation makes the canonical assumption break.
        def chain_greedy(order, first_frag=False):
            seq=[]; cur=start
            for pos,k in enumerate(order):
                li=L[k]
                if pos==0 and first_frag:
                    cand=[]
                    for s in li['lane']:
                        cand.append((abs(cur[along]-a_lo(s)),True)); cand.append((abs(cur[along]-a_hi(s)),False))
                    _,up=min(cand,key=lambda z:z[0])
                else:
                    up=abs(cur[along]-li['lo'])<=abs(cur[along]-li['hi'])
                part=emit(li,up); seq+=part; cur=part[-1][1]
            return seq
        idxs=list(range(n))
        for o in (idxs[:], idxs[::-1], long_ids+short_ids, short_ids[::-1]+long_ids[::-1]):
            candidates.append(chain_greedy(o, False))
            candidates.append(chain_greedy(o, True))

        best=None; best_sc=None
        for seq in candidates:
            if len(seq)!=len(swaths): continue
            seq2 = fix_start(seq)
            sc=score(seq2)
            if best_sc is None or sc<best_sc: best_sc=sc; best=seq2
        return best
    except Exception:  # noqa: BLE001 — 任何异常回退原弓字, 绝不挡覆盖
        return None



def _order_swaths_boustrophedon(
        swaths: Sequence[Swath], start: Point, axis: str,
        spacing_m: float) -> List[Swath]:
    """#3 弓字形排序: 按行号(垂直坐标)聚类成行 → 相邻行连续走、每行交替方向.

    比贪心最近邻可预测、连接段更短、重复更少 —— 尤其大场地/障碍把行打断成
    碎段时, 最近邻会满屋乱跳, 弓字形不会. 退化/异常时回退到最近邻.
    """
    if not swaths:
        return []
    try:
        perp = 1 if axis == "x" else 0   # 垂直扫描方向的坐标 (= 行号方向)
        along = 0 if axis == "x" else 1  # 沿扫描方向的坐标
        tol = max(1e-6, float(spacing_m) * 0.5)

        def perp_of(s):
            return 0.5 * (s[0][perp] + s[1][perp])

        # 1) 按 perpendicular 聚类成"行"
        items = sorted(swaths, key=perp_of)
        rows: List[List[Swath]] = [[items[0]]]
        for s in items[1:]:
            if abs(perp_of(s) - perp_of(rows[-1][-1])) <= tol:
                rows[-1].append(s)
            else:
                rows.append([s])

        # 2) 行的推进方向: 从离机器人最近的那一头开始
        if (abs(perp_of(rows[-1][0]) - start[perp])
                < abs(perp_of(rows[0][0]) - start[perp])):
            rows.reverse()

        # [1939实测 跳带掉头] 隔行清扫: 去程扫偶数行(0,2,4…), 回程扫奇数行(…5,3,1) →
        # 相邻执行行间距 = 2×扫带间距(1.05m), 掉头半径翻倍: wz_max=0.4 时过弯线速度
        # v=wz_max×r 从 ~0.10 提到 ~0.21m/s(转弯丝滑且快, 停顿减少)。副作用全正:
        # 路径折返段空间间距 1.05m > 一切吸附/匹配窗(0.85), 近邻车道歧义根治。
        # 行数<3 不值得(多跑连接段反亏)。末端一次单间距掉头不可避免。
        # [1939实测 跳带掉头] 隔行清扫: 去程扫偶数行(0,2,4…), 回程扫奇数行(…5,3,1) →
        # 相邻执行行间距=2×扫带间距(1.05m), 掉头半径翻倍治急转/减停顿。
        # [2026-07-07二次] 用户拍板顺序扫(从左到右依次)。0.87离墙下安全: 7月2日前顺序+0.55离墙
        # 跑了数周无撞, 上次撞墙主因是0.65离墙(已回退0.87)。拐角会慢(0.525紧掉头), 上机观察。
        if len(rows) >= 10:
            rows = rows[0::2] + rows[1::2][::-1]

        # 3) 逐行: 行内段按 along 排序, 整行方向逐行交替 (弓字形), 端点摆正
        ordered: List[Swath] = []
        forward: Optional[bool] = None
        for r in rows:
            r_sorted = sorted(r, key=lambda s: min(s[0][along], s[1][along]))
            if forward is None:
                lo = min(min(s[0][along], s[1][along]) for s in r_sorted)
                hi = max(max(s[0][along], s[1][along]) for s in r_sorted)
                forward = abs(start[along] - lo) <= abs(start[along] - hi)
            seq = r_sorted if forward else list(reversed(r_sorted))
            for s in seq:
                a, b = s
                if forward and a[along] > b[along]:
                    a, b = b, a
                elif (not forward) and a[along] < b[along]:
                    a, b = b, a
                ordered.append((a, b))
            forward = not forward
        if len(ordered) == len(swaths):
            return ordered
    except Exception:  # noqa: BLE001
        pass
    return _order_swaths_nearest(swaths, start)   # 兜底


def _split_start_swath(
    swaths: Sequence[Swath],
    start: Point,
    min_part_m: float,
    max_lateral_m: float = 0.80,
    start_heading: Optional[float] = None,
) -> Tuple[List[Swath], List[Swath]]:
    """Allow cleaning to begin near the robot instead of forcing an endpoint U-turn.

    Returns ``(start_parts, remaining)``. ``start_parts`` is the split halves of
    the swath the robot sits on, sorted so the yaw-preferred half is first (empty
    if no split happened). The caller keeps that half at the front of the route so
    the boustrophedon orderer can't discard the yaw-aware start decision.
    """
    best_i = -1
    best_proj: Optional[Point] = None
    best_d = float("inf")
    for i, (a, b) in enumerate(swaths):
        proj, t = _project_point_to_segment(start, a, b)
        if t <= 0.05 or t >= 0.95:
            continue
        lateral = _dist(start, proj)
        if lateral < best_d and lateral <= max_lateral_m:
            best_i = i
            best_proj = proj
            best_d = lateral
    if best_i < 0 or best_proj is None:
        return [], list(swaths)

    remaining = list(swaths)
    a, b = remaining.pop(best_i)
    entry = start if best_d <= 0.15 else best_proj
    parts: List[Swath] = []
    for part in ((entry, a), (entry, b)):
        if _dist(part[0], part[1]) >= min_part_m:
            parts.append(part)
    if len(parts) < 2:
        return [], list(swaths)
    if start_heading is not None:
        hx = math.cos(start_heading)
        hy = math.sin(start_heading)

        def heading_score(swath: Swath) -> float:
            vx = swath[1][0] - swath[0][0]
            vy = swath[1][1] - swath[0][1]
            n = math.hypot(vx, vy)
            if n <= 1e-6:
                return -1.0
            return (vx / n) * hx + (vy / n) * hy

        parts.sort(key=lambda s: (-heading_score(s), _dist(s[0], s[1])))
    else:
        parts.sort(key=lambda s: _dist(s[0], s[1]))
    return parts, remaining


def _half_arc(a, b, bulge_dir, mask, resolution, origin_x, origin_y):
    """a→b 半圆弧, 拱向 bulge_dir(单位向量)一侧; 采样全过 mask 校验, 否则 None。"""
    g = _dist(a, b)
    if g < 1e-6:
        return None
    c = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    r = g / 2.0
    a0 = math.atan2(a[1] - c[1], a[0] - c[0])
    n = max(6, int(math.ceil(math.pi * r / 0.05)))
    for sign in (1.0, -1.0):
        pts = []
        ok = True
        for k in range(1, n + 1):
            th = a0 + sign * math.pi * k / n
            p = (c[0] + r * math.cos(th), c[1] + r * math.sin(th))
            if not _point_is_free(mask, p, resolution, origin_x, origin_y):
                ok = False
                break
            pts.append(p)
        if not ok:
            continue
        mid = pts[len(pts) // 2]
        if (mid[0] - c[0]) * bulge_dir[0] + (mid[1] - c[1]) * bulge_dir[1] > 0.0:
            return pts
    return None


def _semicircle_connector(cur, nxt, dir_prev, mask, resolution,
                          origin_x, origin_y, path_step_m, dir_next=None):
    """[2026-07-07 半圆掉头] 并排反向车道 → 半圆弧; [S形双半圆] 下带起点在【身后】
    (墙凸起裁短车道的折返) → 顶部半圆到中线+直下+底部半圆进下带, 全程无尖角。
    不适用/被mask挡 → None(回落A*+圆角)。"""
    dxn, dyn = nxt[0] - cur[0], nxt[1] - cur[1]
    g = math.hypot(dxn, dyn)
    if g < 0.15 or g > 4.0:
        return None
    lon = dxn * dir_prev[0] + dyn * dir_prev[1]
    latx, laty = dxn - lon * dir_prev[0], dyn - lon * dir_prev[1]
    lat = math.hypot(latx, laty)
    if lat < 0.15 or lat > 1.2:
        return None
    if g <= 1.2 and abs(lon) <= 0.35 * g:
        if dir_next is not None and (dir_next[0] * dir_prev[0]
                                     + dir_next[1] * dir_prev[1]) > -0.7:
            return None   # [codex审B1] 下带非反向 → 半圆出口切向不匹配, 回落A*
        arc = _half_arc(cur, nxt, dir_prev, mask, resolution, origin_x, origin_y)
        return ([cur] + arc) if arc else None
    if lon < -0.2:
        _dnp = (dir_next[0] * dir_prev[0] + dir_next[1] * dir_prev[1]
                ) if dir_next is not None else 1.0
        if _dnp < -0.7:
            # [J形 codex审B1后补] 下带【反向】且起点在身后(错层反向车道): 近端半圆
            # (cur→下带延长线上同高点Q, 拱向行进方向外侧) + 沿下带延长线直行到 nxt——
            # 直行段与下带共线同向, 切向完全吻合, 零cusp。
            Q = (cur[0] + latx, cur[1] + laty)
            arcj = _half_arc(cur, Q, dir_prev, mask, resolution, origin_x, origin_y)
            if arcj:
                legj = _densify(Q, nxt, 0.05)
                if all(_point_is_free(mask, q, resolution, origin_x, origin_y)
                       for q in legj):
                    return [cur] + arcj + legj[1:]
            return None
        if _dnp < 0.7:
            return None   # [codex审B1] 方向不明确 → 回落A*
        # S形双半圆: cur→P1(横移一半, 顶部半圆拱向前方) → 中线直行 → P2→nxt(底部
        # 半圆拱向后方)。P1P2在两车道正中线, 与两带各距 lat/2, 无共线折返尖。
        P1 = (cur[0] + latx * 0.5, cur[1] + laty * 0.5)
        P2 = (nxt[0] - latx * 0.5, nxt[1] - laty * 0.5)
        arc1 = _half_arc(cur, P1, dir_prev, mask, resolution, origin_x, origin_y)
        if not arc1:
            return None
        arc2 = _half_arc(P2, nxt, (-dir_prev[0], -dir_prev[1]),
                         mask, resolution, origin_x, origin_y)
        if not arc2:
            return None
        leg = _densify(P1, P2, 0.05)
        for q in leg:
            if not _point_is_free(mask, q, resolution, origin_x, origin_y):
                return None
        return [cur] + arc1 + leg[1:] + arc2
    return None


def _stitch_path(
    free_mask: np.ndarray,
    entry_free_mask: np.ndarray,
    swaths: Sequence[Swath],
    start: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    penalty_mask: Optional[np.ndarray] = None,
    semicircle_uturns: bool = False,  # [2026-07-07 半圆掉头] 并排反向车道间用半圆弧连接
    avoid_traversed_lanes: bool = False,
) -> List[Point]:
    path: List[Point] = [start]
    cur = start
    prev_dir = None
    traversed_penalty = (
        np.asarray(penalty_mask, dtype=bool).copy()
        if penalty_mask is not None
        else np.zeros_like(free_mask, dtype=bool))
    for i, (a, b) in enumerate(swaths):
        # Every inter-swath connector may use the robot-traversable component.
        # Fill-safe swaths can be separated by a narrow doorway where cleaning
        # lanes are intentionally absent; restricting later connectors to the
        # fill mask silently discarded otherwise reachable rooms.
        connector_mask = entry_free_mask
        connector = None
        active_penalty = traversed_penalty.copy()
        if active_penalty.any():
            # Adjacent boustrophedon lanes necessarily meet inside one small
            # headland.  Clear only their two endpoints; elsewhere a connector
            # pays a soft cost for crossing an already completed lane.
            endpoint_clear_cells = max(
                1, int(math.ceil(0.40 / resolution)))
            for endpoint in (cur, a):
                endpoint_row, endpoint_col = _world_to_cell(
                    endpoint, resolution, origin_x, origin_y)
                active_penalty[
                    max(0, endpoint_row - endpoint_clear_cells):
                    min(active_penalty.shape[0],
                        endpoint_row + endpoint_clear_cells + 1),
                    max(0, endpoint_col - endpoint_clear_cells):
                    min(active_penalty.shape[1],
                        endpoint_col + endpoint_clear_cells + 1),
                ] = False
        if semicircle_uturns and i > 0 and prev_dir is not None:
            # [半圆掉头] 用户实测: 三角切角掉头MPPI转不过→冻结→前跳→偏离整单停。
            # 半圆拱向车道末端外侧(headland 0.34带, r=间距/2≈0.26装得下), 被挡回落
            # 掩码内Theta*，Theta*异常/无解再回落A*。
            # [codex审B1] 传下一带方向: 半圆出口切向=-dir_prev(须下带反向), S形出口
            # =+dir_prev(须下带同向), 不校验会在错峰车道处造180°共线折返cusp。
            _dnab = _dist(a, b)
            _dir_next = (((b[0] - a[0]) / _dnab, (b[1] - a[1]) / _dnab)
                         if _dnab > 1e-6 else None)
            connector = _semicircle_connector(
                cur, a, prev_dir, free_mask, resolution, origin_x, origin_y,
                path_step_m, dir_next=_dir_next)
        if connector is None:
            connector = _connect_points(
                connector_mask, cur, a, resolution, origin_x, origin_y, path_step_m,
                penalty_mask=(active_penalty
                              if active_penalty.any() else None),
                prefer_theta=True)
        if not connector:
            return []
        _extend_dedup(path, connector[1:])
        body = _densify(a, b, path_step_m)
        _extend_dedup(path, body[1:] if body and path else body)
        cur = b
        _d_ab = _dist(a, b)
        if _d_ab > 1e-6:
            prev_dir = ((b[0] - a[0]) / _d_ab, (b[1] - a[1]) / _d_ab)
        if avoid_traversed_lanes:
            traversed_penalty |= _polyline_cleaning_mask(
                free_mask.shape,
                connector + _densify(a, b, path_step_m)[1:],
                resolution, origin_x, origin_y,
                max(resolution, 0.18),
            )
    return path


def _rdp_masked(
    path: Sequence[Point],
    eps_m: float,
    free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> List[Point]:
    """[0703晨 平滑] Douglas-Peucker 简化 + 自由空间弦校验: 把 A*缝合段/栅格描迹的
    密集45°/90°楼梯坍缩成直弦。楼梯腿长≈格距, 恒小于 _smooth_path_corners 圆角环节的
    trim 门槛(path_step*0.75), 原来整段跳过 → 规划轨迹肉眼锯齿 + MPPI 循迹蛇形的主源。
    弦不过 free_mask 校验 → 在最大偏差点分裂重试, 最终退化为原点列(零安全回退)。"""
    pts = [tuple(p) for p in path]
    n = len(pts)
    if n < 3:
        return list(pts)
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        ax, ay = pts[i0]
        bx, by = pts[i1]
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        dmax, imax = -1.0, -1
        for i in range(i0 + 1, i1):
            px, py = pts[i]
            if seg_len < 1e-9:
                d = math.hypot(px - ax, py - ay)
            else:
                d = abs(dx * (py - ay) - dy * (px - ax)) / seg_len
            if d > dmax:
                dmax, imax = d, i
        if dmax > eps_m or not _world_segment_is_free(
                free_mask, pts[i0], pts[i1], resolution, origin_x, origin_y):
            keep.add(imax)
            stack.append((i0, imax))
            stack.append((imax, i1))
    return [pts[i] for i in sorted(keep)]


def _smooth_path_corners(
    path: Sequence[Point],
    free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    turn_radius_m: float,
    path_step_m: float,
) -> List[Point]:
    if len(path) < 3 or turn_radius_m <= 0.0:
        return list(path)
    # Simplify collinear points FIRST, THEN remove reversal spurs. Running spur
    # removal on the still-densified path deletes genuine swath-entry corners,
    # leaving two ultra-short segments the rounding loop then skips
    # (trim < path_step_m*0.75) — an unroundable ~73° kink. Simplifying first
    # keeps the real corner as a clean vertex the rounder can round.
    simplified = _rdp_masked(path, 0.04, free_mask, resolution,
                             origin_x, origin_y)   # [0703晨] 楼梯坍缩(弦偏差≤4cm)
    simplified = _simplify_collinear(simplified)
    simplified = _remove_reversal_spurs(
        simplified, free_mask, resolution, origin_x, origin_y,
        max_shortcut_m=max(turn_radius_m * 2.0, path_step_m * 3.0),
    )
    rounded: List[Point] = [simplified[0]]
    min_turn_rad = math.radians(15.0)

    for i in range(1, len(simplified) - 1):
        a = simplified[i - 1]
        b = simplified[i]
        c = simplified[i + 1]
        vin = (b[0] - a[0], b[1] - a[1])
        vout = (c[0] - b[0], c[1] - b[1])
        lin = math.hypot(*vin)
        lout = math.hypot(*vout)
        if lin < 1e-6 or lout < 1e-6:
            _extend_dedup(rounded, [b])
            continue
        din = (vin[0] / lin, vin[1] / lin)
        dout = (vout[0] / lout, vout[1] / lout)
        dot = max(-1.0, min(1.0, din[0] * dout[0] + din[1] * dout[1]))
        turn = math.acos(dot)
        trim = min(turn_radius_m, lin * 0.45, lout * 0.45)
        if turn < min_turn_rad or trim < path_step_m * 0.75:
            _extend_dedup(rounded, [b])
            continue

        accepted_curve: Optional[Tuple[Point, List[Point]]] = None
        for scale in (1.0, 0.70, 0.50, 0.35):
            scaled_trim = trim * scale
            if scaled_trim < path_step_m * 0.50:
                continue
            start = (b[0] - din[0] * scaled_trim, b[1] - din[1] * scaled_trim)
            end = (b[0] + dout[0] * scaled_trim, b[1] + dout[1] * scaled_trim)
            curve = _quadratic_bezier(start, b, end, max(path_step_m * 0.5, 0.03))
            if all(_point_is_free(free_mask, p, resolution, origin_x, origin_y)
                   for p in curve):
                accepted_curve = (start, curve)
                break
        if accepted_curve is None:
            _extend_dedup(rounded, [b])
        else:
            start, curve = accepted_curve
            _extend_dedup(rounded, [start])
            _extend_dedup(rounded, curve[1:])
    _extend_dedup(rounded, [simplified[-1]])
    dense = _densify_polyline(rounded, path_step_m)
    dense = _remove_reversal_spurs(
        dense, free_mask, resolution, origin_x, origin_y,
        max_shortcut_m=max(turn_radius_m * 2.0, path_step_m * 3.0),
    )
    return _densify_polyline(dense, path_step_m)


def _remove_reversal_spurs(
    path: Sequence[Point],
    free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    max_turn_rad: float = math.radians(95.0),
    max_shortcut_m: float = 0.60,
) -> List[Point]:
    """Drop tiny out-and-back cusps that make MPPI crawl at coverage turns."""
    pts = list(path)
    if len(pts) < 3:
        return pts

    for _ in range(8):
        changed = False
        out: List[Point] = [pts[0]]
        i = 1
        while i < len(pts) - 1:
            a = out[-1]
            b = pts[i]
            c = pts[i + 1]
            turn = _turn_angle(a, b, c)
            shortcut_len = _dist(a, c)
            if (
                turn >= max_turn_rad
                and shortcut_len <= max_shortcut_m
                and _world_segment_is_free(
                    free_mask, a, c, resolution, origin_x, origin_y)
            ):
                changed = True
                i += 1
                continue
            _extend_dedup(out, [b])
            i += 1
        _extend_dedup(out, [pts[-1]])
        pts = out
        if not changed:
            break
    return pts


def _turn_angle(a: Point, b: Point, c: Point) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.acos(dot)


def _world_segment_is_free(
    free_mask: np.ndarray,
    a: Point,
    b: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> bool:
    d = _dist(a, b)
    if d <= 1e-6:
        return _point_is_free(free_mask, a, resolution, origin_x, origin_y)
    samples = max(2, int(math.ceil(d / max(resolution * 0.5, 0.02))) + 1)
    for i in range(samples):
        t = i / (samples - 1)
        p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if not _point_is_free(free_mask, p, resolution, origin_x, origin_y):
            return False
    return True


def _polyline_is_free(
    free_mask: np.ndarray,
    path: Sequence[Point],
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> bool:
    if not path:
        return False
    if len(path) == 1:
        return _point_is_free(free_mask, path[0], resolution, origin_x, origin_y)
    return all(
        _world_segment_is_free(free_mask, a, b, resolution, origin_x, origin_y)
        for a, b in zip(path, path[1:])
    )


def _simplify_collinear(path: Sequence[Point], turn_epsilon_rad: float = math.radians(3.0)
                        ) -> List[Point]:
    if len(path) < 3:
        return list(path)
    out: List[Point] = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        if math.acos(dot) >= turn_epsilon_rad:
            out.append(b)
    out.append(path[-1])
    return out


def _quadratic_bezier(a: Point, ctrl: Point, b: Point, step_m: float) -> List[Point]:
    approx_len = _dist(a, ctrl) + _dist(ctrl, b)
    n = max(4, int(math.ceil(approx_len / max(0.02, step_m))) + 1)
    pts: List[Point] = []
    for i in range(n):
        t = i / (n - 1)
        omt = 1.0 - t
        pts.append((
            omt * omt * a[0] + 2.0 * omt * t * ctrl[0] + t * t * b[0],
            omt * omt * a[1] + 2.0 * omt * t * ctrl[1] + t * t * b[1],
        ))
    return pts


def _densify_polyline(path: Sequence[Point], step_m: float) -> List[Point]:
    if not path:
        return []
    out: List[Point] = [path[0]]
    for a, b in zip(path, path[1:]):
        seg = _densify(a, b, step_m)
        _extend_dedup(out, seg[1:])
    return out


def _connect_points(
    free_mask: np.ndarray,
    start: Point,
    goal: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
    penalty_mask: Optional[np.ndarray] = None,
    prefer_theta: bool = False,
) -> List[Point]:
    """Connect two world points without leaving ``free_mask``.

    Direct line-of-sight is always preferred.  Offline coverage assembly opts
    into mask-constrained Theta* for a shorter any-angle connector; if Theta*
    cannot produce a world-space-valid polyline we retry the established A*
    implementation.  ``prefer_theta`` defaults to False deliberately so the
    online perimeter recovery caller keeps its previous behaviour.
    """
    start_cell = _snap_to_free(
        free_mask, _world_to_cell(start, resolution, origin_x, origin_y))
    goal_cell = _snap_to_free(
        free_mask, _world_to_cell(goal, resolution, origin_x, origin_y))
    if start_cell is None or goal_cell is None:
        return []
    _direct_ok = (
        _line_in_free(free_mask, start_cell, goal_cell)
        and _world_segment_is_free(
            free_mask, start, goal, resolution, origin_x, origin_y)
    )
    # [1052] 有借道诉求且直线要穿已扫区 → 不抄近道, 交给加权A*看有没有值得的未扫绕线
    if (_direct_ok and penalty_mask is not None
            and not _line_in_free(np.logical_not(penalty_mask),
                                  start_cell, goal_cell)):
        _direct_ok = False
    if _direct_ok:
        return _densify(start, goal, path_step_m)

    if prefer_theta:
        theta_cells = _theta_star(
            free_mask, start_cell, goal_cell, penalty_mask=penalty_mask)
        if theta_cells:
            theta_pts = [
                _cell_to_world(c, resolution, origin_x, origin_y)
                for c in theta_cells
            ]
            theta_pts[0] = start
            theta_pts[-1] = goal
            if _polyline_is_free(
                    free_mask, theta_pts, resolution, origin_x, origin_y):
                # Theta* intentionally returns only turning points.  Restore the
                # coverage planner's <= path_step_m waypoint contract before the
                # later common corner-smoothing pass.
                return _densify_polyline(theta_pts, path_step_m)

    # Safety/compatibility fallback: preserve the old grid A* connector.
    cells = _astar(free_mask, start_cell, goal_cell, penalty_mask)
    if not cells:
        return []
    pts = [_cell_to_world(c, resolution, origin_x, origin_y) for c in cells]
    if pts:
        pts[0] = start
        pts[-1] = goal
    if not _polyline_is_free(free_mask, pts, resolution, origin_x, origin_y):
        return []
    return pts


def _supercover_line_cells(start: Cell, goal: Cell) -> List[Cell]:
    """Return every grid cell touched by the centre-to-centre segment.

    When a segment crosses an exact cell corner both orthogonal side cells are
    included.  Requiring both to be free prevents an any-angle connector from
    squeezing diagonally between two inflated obstacles.
    """
    y0, x0 = start
    y1, x1 = goal
    nx = abs(x1 - x0)
    ny = abs(y1 - y0)
    sx = 0 if x0 == x1 else (1 if x1 > x0 else -1)
    sy = 0 if y0 == y1 else (1 if y1 > y0 else -1)
    x, y = x0, y0
    ix = iy = 0
    cells: List[Cell] = [(y, x)]

    def append(cell: Cell):
        if cells[-1] != cell:
            cells.append(cell)

    while ix < nx or iy < ny:
        # Compare the parametric distance to the next vertical/horizontal grid
        # boundary using integers, avoiding fragile floating-point equality.
        cross_x = (1 + 2 * ix) * ny
        cross_y = (1 + 2 * iy) * nx
        if cross_x < cross_y:
            x += sx
            ix += 1
        elif cross_x > cross_y:
            y += sy
            iy += 1
        else:
            # Exact corner crossing: the diagonal is legal only when the two
            # side cells are also free (same safety rule as the old 8-neighbour
            # A* implementation).
            append((y, x + sx))
            append((y + sy, x))
            x += sx
            y += sy
            ix += 1
            iy += 1
        append((y, x))
    return cells


def _theta_segment_cost(
    free_mask: np.ndarray,
    start: Cell,
    goal: Cell,
    penalty_mask: Optional[np.ndarray] = None,
    penalty: float = 3.0,
) -> float:
    """Line-of-sight distance, or infinity when the segment leaves the mask."""
    length = math.hypot(goal[0] - start[0], goal[1] - start[1])
    if not _supercover_line_is_free(free_mask, start, goal):
        return float("inf")
    if length <= 1e-12 or penalty_mask is None:
        return length

    # Preserve the existing "already-cleaned cells cost more" semantics for
    # directed reclean connectors.  The LOS safety test above remains strict;
    # this average only ranks otherwise valid Theta* segments.
    cells = _supercover_line_cells(start, goal)
    traversed = cells[1:] or cells
    weight = sum(
        penalty if bool(penalty_mask[y, x]) else 1.0
        for y, x in traversed
    ) / len(traversed)
    return length * weight


def _supercover_line_is_free(
    free_mask: np.ndarray,
    start: Cell,
    goal: Cell,
) -> bool:
    """Allocation-free conservative line-of-sight check for Theta*."""
    height, width = free_mask.shape

    def is_free(y: int, x: int) -> bool:
        return 0 <= y < height and 0 <= x < width and bool(free_mask[y, x])

    y0, x0 = start
    y1, x1 = goal
    if not is_free(y0, x0):
        return False
    nx = abs(x1 - x0)
    ny = abs(y1 - y0)
    sx = 0 if x0 == x1 else (1 if x1 > x0 else -1)
    sy = 0 if y0 == y1 else (1 if y1 > y0 else -1)
    x, y = x0, y0
    ix = iy = 0
    while ix < nx or iy < ny:
        cross_x = (1 + 2 * ix) * ny
        cross_y = (1 + 2 * iy) * nx
        if cross_x < cross_y:
            x += sx
            ix += 1
        elif cross_x > cross_y:
            y += sy
            iy += 1
        else:
            if not (is_free(y, x + sx) and is_free(y + sy, x)):
                return False
            x += sx
            y += sy
            ix += 1
            iy += 1
        if not is_free(y, x):
            return False
    return True


def _theta_star(
    free_mask: np.ndarray,
    start: Cell,
    goal: Cell,
    penalty_mask: Optional[np.ndarray] = None,
    penalty: float = 3.0,
) -> List[Cell]:
    """Lazy Theta* constrained to ``free_mask`` with conservative LOS.

    The returned sequence contains start, goal and only the required turning
    cells.  Lazy Theta* defers each expensive line-of-sight check until a cell is
    expanded, which matters on Jetson-sized floor grids.  Weighted reclean
    connectors deliberately return no Theta* result so ``_connect_points`` uses
    the established weighted A* fallback without changing its semantics.
    """
    if start == goal:
        return [start]
    if not (_cell_is_free(free_mask, start) and _cell_is_free(free_mask, goal)):
        return []
    if penalty_mask is not None:
        return []

    height, width = free_mask.shape
    parent: Dict[Cell, Cell] = {start: start}
    g_score: Dict[Cell, float] = {start: 0.0}
    closed = set()

    def heuristic(cell: Cell) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap: List[Tuple[float, Cell]] = [(heuristic(start), start)]
    neighbours = (
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue

        # Lazy Theta*: the parent shortcut was optimistic when this cell was
        # queued.  Validate it now; if an obstacle blocks it, repair the vertex
        # from the best already-expanded adjacent cell.
        if not math.isfinite(_theta_segment_cost(
                free_mask, parent[cur], cur, penalty_mask, penalty)):
            best_parent = None
            best_cost = float("inf")
            for dy, dx in neighbours:
                candidate = (cur[0] + dy, cur[1] + dx)
                if candidate not in closed:
                    continue
                edge_cost = _theta_segment_cost(
                    free_mask, candidate, cur, penalty_mask, penalty)
                if not math.isfinite(edge_cost):
                    continue
                candidate_cost = g_score[candidate] + edge_cost
                if candidate_cost < best_cost:
                    best_parent = candidate
                    best_cost = candidate_cost
            if best_parent is None:
                continue
            parent[cur] = best_parent
            g_score[cur] = best_cost

        if cur == goal:
            out = [cur]
            # Every non-start node receives a parent on relaxation.  Keep a
            # guard so a corrupted/cyclic parent map fails closed to A*.
            for _ in range(len(parent) + 1):
                if out[-1] == start:
                    return list(reversed(out))
                nxt_parent = parent.get(out[-1])
                if nxt_parent is None or nxt_parent == out[-1]:
                    return []
                out.append(nxt_parent)
            return []
        closed.add(cur)

        for dy, dx in neighbours:
            nxt = (cur[0] + dy, cur[1] + dx)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if not free_mask[nxt[0], nxt[1]] or nxt in closed:
                continue

            candidate_parent = parent[cur]
            # Visibility is intentionally not checked here; SetVertex above
            # validates/repairs the shortcut once ``nxt`` reaches the queue head.
            segment_cost = math.hypot(
                nxt[0] - candidate_parent[0], nxt[1] - candidate_parent[1])
            tentative = g_score[candidate_parent] + segment_cost

            if tentative + 1e-12 < g_score.get(nxt, float("inf")):
                parent[nxt] = candidate_parent
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nxt), nxt))
    return []


def _astar(free_mask: np.ndarray, start: Cell, goal: Cell,
           penalty_mask: Optional[np.ndarray] = None,
           penalty: float = 3.0) -> List[Cell]:
    """penalty_mask[cell]=True 的格步进代价×penalty —— [1052用户需求] 补漏连接段优先
    借道「未扫区」(顺路补扫), 已扫区非禁行只是代价高(绕行≤3倍长度才值得)。
    启发式仍是未加权欧氏距离(≤真实代价, 可采纳性不变)。"""
    if start == goal:
        return [start]
    height, width = free_mask.shape
    open_heap: List[Tuple[float, Cell]] = []
    heapq.heappush(open_heap, (0.0, start))
    came: Dict[Cell, Optional[Cell]] = {start: None}
    g_score: Dict[Cell, float] = {start: 0.0}

    def heuristic(c: Cell) -> float:
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            out: List[Cell] = []
            node: Optional[Cell] = cur
            while node is not None:
                out.append(node)
                node = came[node]
            return list(reversed(out))
        for dy, dx, cost in (
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ):
            ny, nx = cur[0] + dy, cur[1] + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if not free_mask[ny, nx]:
                continue
            if dy and dx:
                if not (free_mask[cur[0], nx] and free_mask[ny, cur[1]]):
                    continue
            nxt = (ny, nx)
            step = cost * penalty if (penalty_mask is not None
                                      and penalty_mask[ny, nx]) else cost
            tentative = g_score[cur] + step
            if tentative < g_score.get(nxt, float("inf")):
                came[nxt] = cur
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nxt), nxt))
    return []


def _snap_to_free(free_mask: np.ndarray, cell: Cell, max_radius: int = 80
                  ) -> Optional[Cell]:
    height, width = free_mask.shape
    y = max(0, min(height - 1, cell[0]))
    x = max(0, min(width - 1, cell[1]))
    if free_mask[y, x]:
        return (y, x)
    from collections import deque
    q = deque([(y, x)])
    seen = {(y, x)}
    while q:
        cy, cx = q.popleft()
        if abs(cy - y) > max_radius or abs(cx - x) > max_radius:
            continue
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if (ny, nx) in seen:
                continue
            if free_mask[ny, nx]:
                return (ny, nx)
            seen.add((ny, nx))
            q.append((ny, nx))
    return None


def _line_in_free(free_mask: np.ndarray, start: Cell, goal: Cell) -> bool:
    height, width = free_mask.shape
    y0, x0 = start
    y1, x1 = goal
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    y, x = y0, x0
    while True:
        if not (0 <= y < height and 0 <= x < width and free_mask[y, x]):
            return False
        if (y, x) == (y1, x1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _densify(a: Point, b: Point, step_m: float) -> List[Point]:
    d = _dist(a, b)
    n = max(2, int(math.ceil(d / max(0.02, step_m))) + 1)
    return [
        (a[0] + (i / (n - 1)) * (b[0] - a[0]),
         a[1] + (i / (n - 1)) * (b[1] - a[1]))
        for i in range(n)
    ]


def _extend_dedup(path: List[Point], pts: Sequence[Point], min_step: float = 0.015):
    for p in pts:
        if not path or _dist(path[-1], p) >= min_step:
            path.append(p)


def _world_to_cell(p: Point, resolution: float, origin_x: float, origin_y: float
                   ) -> Cell:
    return (int(math.floor((p[1] - origin_y) / resolution)),
            int(math.floor((p[0] - origin_x) / resolution)))


def _cell_to_world(c: Cell, resolution: float, origin_x: float, origin_y: float
                   ) -> Point:
    return origin_x + (c[1] + 0.5) * resolution, origin_y + (c[0] + 0.5) * resolution


def _cell_is_free(free_mask: np.ndarray, cell: Cell) -> bool:
    return (0 <= cell[0] < free_mask.shape[0]
            and 0 <= cell[1] < free_mask.shape[1]
            and bool(free_mask[cell[0], cell[1]]))


def _point_is_free(
    free_mask: np.ndarray,
    p: Point,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> bool:
    return _cell_is_free(free_mask, _world_to_cell(p, resolution, origin_x, origin_y))


def _project_point_to_segment(p: Point, a: Point, b: Point) -> Tuple[Point, float]:
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return a, 0.0
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / denom
    t = max(0.0, min(1.0, t))
    return (a[0] + t * vx, a[1] + t * vy), t


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _trace_perimeter_rings(
    free: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    step_m: float,
    min_ring_cells: int = 8,
) -> List[List[Point]]:
    """沿 `free` 区域边界(外边界 + 内部障碍岛洞)生成有序闭环 polyline(世界系)。

    有向半边法: 每个 free cell 的某 4 邻为非 free 时, 沿该 cell 边加一条有向边(free 在
    前进方向左侧), 顶点用 (row,col) 栅格顶点。用 multimap + 右手定则(右转优先)在对角相触/
    saddle 顶点正确分流, 半边首尾相接串成闭环; 取每条边所属 free cell 的中心(已含
    robot_radius 留边)组成环, 故环点都落在可达自由格内。退化短环由 min_ring_cells 过滤。
    """
    from collections import defaultdict
    h, w = free.shape
    # 有向半边 multimap: start_vertex -> [end_vertex,...]; edge_cell 存每条边所属 free cell。
    # 用 multimap(不是 dict 单值)+ 下面的右手定则, 才能在对角相触/saddle 处正确分流, 不丢边。
    out_edges: Dict[Cell, List[Cell]] = defaultdict(list)
    edge_cell: Dict[Tuple[Cell, Cell], Cell] = {}
    edge_list: List[Tuple[Cell, Cell]] = []

    def _add(sv: Cell, ev: Cell, cell: Cell) -> None:
        out_edges[sv].append(ev)
        edge_cell[(sv, ev)] = cell
        edge_list.append((sv, ev))

    for r in range(h):
        for c in range(w):
            if not free[r, c]:
                continue
            if r == 0 or not free[r - 1, c]:            # North 邻非 free
                _add((r, c + 1), (r, c), (r, c))
            if r == h - 1 or not free[r + 1, c]:        # South
                _add((r + 1, c), (r + 1, c + 1), (r, c))
            if c == 0 or not free[r, c - 1]:            # West
                _add((r, c), (r + 1, c), (r, c))
            if c == w - 1 or not free[r, c + 1]:        # East
                _add((r + 1, c + 1), (r, c + 1), (r, c))

    used: set = set()

    def _pick_next(v: Cell, d_in: Cell) -> Optional[Cell]:
        # 右手定则: 相对来向, 右转 > 直行 > 左转 > 掉头(free 在左侧, 优先贴边右转)。
        cw = (d_in[1], -d_in[0])
        ccw = (-d_in[1], d_in[0])
        rev = (-d_in[0], -d_in[1])
        prio = {cw: 0, d_in: 1, ccw: 2, rev: 3}
        best = None
        best_pri = 99
        for ev in out_edges.get(v, ()):
            if (v, ev) in used:
                continue
            d = (ev[0] - v[0], ev[1] - v[1])
            p = prio.get(d, 4)
            if p < best_pri:
                best_pri, best = p, ev
        return best

    rings: List[List[Point]] = []
    max_iter = len(edge_list) + 5
    for (s0, e0) in edge_list:
        if (s0, e0) in used:
            continue
        loop_cells: List[Cell] = []
        cur_s, cur_e = s0, e0
        guard = 0
        while (cur_s, cur_e) not in used and guard <= max_iter:
            used.add((cur_s, cur_e))
            loop_cells.append(edge_cell[(cur_s, cur_e)])
            d_in = (cur_e[0] - cur_s[0], cur_e[1] - cur_s[1])
            nv = _pick_next(cur_e, d_in)
            if nv is None:
                break
            cur_s, cur_e = cur_e, nv
            guard += 1
        if len(loop_cells) < min_ring_cells:
            continue
        # 去连续重复 cell + 去首尾重复(闭合)
        seq: List[Cell] = []
        for cell in loop_cells:
            if not seq or seq[-1] != cell:
                seq.append(cell)
        if len(seq) > 1 and seq[0] == seq[-1]:
            seq.pop()
        if len(seq) < 3:
            continue
        # [1700实测修复] 切段规则升级: 原"对角跳变一律断段"把斜边楼梯逐台阶剪碎
        # (手画五边形5条斜边 → 67段全<8格 → 全被 min_ring_cells 剔除 → 环数0 → 沿边被
        # silently跳过="清扫区太窄")。对角跳变分两种:
        #   ①斜边楼梯: 某个正交过渡格是 free → 插入过渡格补成 4-邻接, 环保持连续
        #     (过渡格在腐蚀后 mask 内, 不损失离障安全性);
        #   ②真 saddle: 两个正交过渡格都非 free(斜穿障碍角) → 才断段(保留防穿角安全性)。
        def _join(a: Cell, b: Cell) -> Optional[List[Cell]]:
            """a→b 之间需插入的过渡格; 4-邻接=[], 楼梯=[过渡格], 真saddle/远跳=None(断)。"""
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if dr + dc == 1:
                return []
            if dr == 1 and dc == 1:
                for c in ((a[0], b[1]), (b[0], a[1])):
                    if _cell_is_free(free, c):
                        return [c]
            return None

        seq2: List[Cell] = [seq[0]]
        brk: set = set()                     # seq2 中"与前一格断开"的下标
        for a, b in zip(seq, seq[1:]):
            j = _join(a, b)
            if j is None:
                brk.add(len(seq2))
            else:
                for c in j:
                    if seq2[-1] != c:
                        seq2.append(c)
            seq2.append(b)
        wrap = _join(seq2[-1], seq2[0])      # 首尾闭合边(同样允许楼梯过渡); None=闭不上
        segs: List[List[Cell]] = [[seq2[0]]]
        for i in range(1, len(seq2)):
            if i in brk:
                segs.append([seq2[i]])
            else:
                segs[-1].append(seq2[i])
        if wrap is not None and len(segs) >= 2:
            # 环中有真 saddle 断点: 首尾两弧经(过渡格校验过的)闭合边相连, 其余保持开弧,
            # 弧间由 _assemble 的掩码内Theta*连接(A*兜底且均经校验), 不会跨障碍直连。
            segs[0] = segs[-1] + wrap + segs[0]
            segs.pop()
        elif wrap is not None:
            for c in wrap:
                segs[0].append(c)
            segs[0].append(segs[0][0])       # 完整单段环闭合
        for seg in segs:
            if len(seg) < min_ring_cells:
                continue
            pts = [_cell_to_world(c, resolution, origin_x, origin_y) for c in seg]
            pts = _simplify_collinear(pts)
            dense = _densify_polyline(pts, step_m)
            if len(dense) >= 2:
                rings.append(dense)
    return rings


def _assemble_perimeter_path(
    start: Point,
    rings: Sequence[Sequence[Point]],
    free_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    path_step_m: float,
) -> List[Point]:
    """把若干沿边环/弧按"就近"贪心串成一条 polyline, 从 `start` 出发。
    每段内部保持原样(已是 4-邻接、densify 过, 不再重连, 否则破坏贴边覆盖);
    段与段之间用掩码内Theta*连接、A*兜底(均校验过)。环是已闭合的 polyline,
    弧是开的, 统一从更近端点进入。"""
    remaining = [list(r) for r in rings if len(r) >= 2]
    out: List[Point] = []
    cur = start
    while remaining:
        best_i = 0
        best_rev = False
        best_d = float("inf")
        for i, poly in enumerate(remaining):
            d0 = _dist(cur, poly[0])
            d1 = _dist(cur, poly[-1])
            if d0 < best_d:
                best_d, best_i, best_rev = d0, i, False
            if d1 < best_d:
                best_d, best_i, best_rev = d1, i, True
        poly = remaining.pop(best_i)
        # [2026-07-08 沿边就近起步] 闭合环: 旋转到离 cur 最近的点起步(而非固定描迹起点)。
        # 用户确认逻辑: 沿边起点 = 分界线上离【填充结束点】最近的点(cur=填充结束车位/上段出口)。
        # 原来闭环只在 poly[0]/poly[-1](=同一描迹起点)进入 → 沿边从固定点起、再从填充结束
        # 点拉长连接段过去空跑。旋转后连接段 = 填充结束→环上最近点(最短)。
        if len(poly) > 4 and _dist(poly[0], poly[-1]) <= path_step_m * 2.5:
            _k = min(range(len(poly)), key=lambda _i: _dist(cur, poly[_i]))
            poly = poly[_k:] + poly[1:_k + 1]
        elif best_rev:
            poly = list(reversed(poly))
        connector = _connect_points(
            free_mask, cur, poly[0], resolution, origin_x, origin_y, path_step_m,
            prefer_theta=True)
        if not connector:
            # 连不到该段(被障碍隔开) → 跳过, 绝不直线 teleport 跨障碍。下一轮选其它段。
            continue
        _extend_dedup(out, connector)
        _extend_dedup(out, poly)
        cur = poly[-1]
    return out


def order_swaths_unified(swaths, start, angle_rad, spacing_m):
    """[0703 B1-lite] 跨区域扫带统一排序: 旋到主方向帧整体弓字(含跳带隔行), 端点回世界系。
    纯几何(扫带本身已由各区域mask裁剪合法); 缝合/连接段避障由调用方 _stitch_path 负责。
    治"补漏一块一个方向互相乱穿": 所有碎块的带按同一主方向像一张大弓字一样排。"""
    if not swaths:
        return []
    th = float(angle_rad or 0.0)
    ct, st = math.cos(th), math.sin(th)

    def to_r(p):
        return (ct * p[0] + st * p[1], -st * p[0] + ct * p[1])

    def to_w(q):
        return (ct * q[0] - st * q[1], st * q[0] + ct * q[1])

    rsw = [(to_r(a), to_r(b)) for a, b in swaths]
    rord = _order_swaths_boustrophedon(rsw, to_r(start), "x", spacing_m)
    return [(to_w(a), to_w(b)) for a, b in rord]


def coverage_swath_spacing(clean_width_m: float, overlap: float = 0.875) -> float:
    """扫带间距 = 清扫宽度 * 重叠系数。注意: 用清扫宽度, 不是导航安全半径。"""
    if clean_width_m <= 0:
        raise ValueError("clean_width_m must be > 0")
    return clean_width_m * overlap
