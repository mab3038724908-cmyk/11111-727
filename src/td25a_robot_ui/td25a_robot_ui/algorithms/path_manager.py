"""Stateful rolling-window management for long coverage paths.

The coverage planner owns the immutable logical route.  :class:`PathManager`
keeps that route, projects the robot onto it without jumping across adjacent
lanes, and exposes short arc-length windows for execution and MPPI debugging.
It is deliberately ROS-free so the geometry and state machine can be tested
off-robot.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


Point2 = Tuple[float, float]


class ExecutionState(str, Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    BLOCKED = "BLOCKED"
    RECOVERING = "RECOVERING"
    REPLANNING = "REPLANNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PathManagerConfig:
    execution_mode: str = "rolling_window_mode"
    window_length: float = 50.0
    min_window_length: float = 20.0
    max_window_length: float = 60.0
    mppi_reference_length: float = 5.0
    mppi_reference_min_length: float = 5.0
    mppi_reference_max_length: float = 5.0
    mppi_prediction_horizon: float = 7.968
    mppi_safety_margin: float = 0.5
    mppi_costmap_radius: float = 5.5
    mppi_costmap_margin: float = 0.5
    backtrack_margin: float = 1.0
    progress_search_forward_distance: float = 4.0
    progress_search_backward_distance: float = 0.75
    max_heading_error: float = math.radians(80.0)
    max_cross_track_error: float = 1.0
    progress_jump_confirm_distance: float = 0.80
    progress_jump_confirm_frames: int = 2
    segment_transition_margin: float = 0.45
    blocked_timeout: float = 8.0
    blocked_velocity_threshold: float = 0.05
    blocked_angular_threshold: float = 0.08
    blocked_progress_threshold: float = 0.20
    obstacle_confirmation_time: float = 3.0
    path_update_rate: float = 5.0
    path_update_distance: float = 0.75
    min_window_remaining_length: float = 18.0
    window_length_change_rate: float = 8.0
    reconnect_min_distance: float = 3.0
    reconnect_max_distance: float = 15.0
    skipped_segment_max_length: float = 3.0
    path_point_spacing: float = 0.10
    coverage_overlap: float = 0.175
    goal_tolerance: float = 0.12
    min_goal_endpoint_distance: float = 0.75

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PathManagerConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in values if key in known})

    def validate(self) -> None:
        if self.execution_mode not in {"fixed_segment_mode", "rolling_window_mode"}:
            raise ValueError("execution_mode must be fixed_segment_mode or rolling_window_mode")
        if not 0.0 < self.min_window_length <= self.window_length <= self.max_window_length:
            raise ValueError("window lengths must satisfy 0 < min <= default <= max")
        if self.backtrack_margin < 0.0:
            raise ValueError("backtrack_margin must be non-negative")
        if self.progress_search_forward_distance <= 0.0:
            raise ValueError("progress_search_forward_distance must be positive")
        if self.progress_search_backward_distance < 0.0:
            raise ValueError("progress_search_backward_distance must be non-negative")
        if self.path_update_rate <= 0.0 or self.path_update_distance <= 0.0:
            raise ValueError("path update rate and distance must be positive")
        if self.progress_jump_confirm_frames < 1:
            raise ValueError("progress_jump_confirm_frames must be >= 1")
        if self.min_goal_endpoint_distance <= self.goal_tolerance:
            raise ValueError("min_goal_endpoint_distance must exceed goal_tolerance")


@dataclass(frozen=True)
class ManagedPose:
    x: float
    y: float
    yaw: float
    s: float


@dataclass(frozen=True)
class PathProjection:
    index: int
    segment_index: int
    segment_id: int
    s: float
    x: float
    y: float
    cross_track_error: float
    heading_error: float
    confirmed: bool


@dataclass(frozen=True)
class PathWindow:
    poses: Tuple[ManagedPose, ...]
    start_s: float
    end_s: float
    start_index: int
    end_index: int
    path_version: int
    remaining_length: float


@dataclass(frozen=True)
class PathRange:
    start_s: float
    end_s: float
    reason: str = ""


@dataclass(frozen=True)
class BlockageObservation:
    blocked: bool
    stalled_for: float
    progress_delta: float
    reason: str


class BlockageDetector:
    """Multi-signal blockage detector independent from instantaneous speed."""

    def __init__(self, config: PathManagerConfig):
        self.config = config
        self.reset()

    def reset(self, now: float = 0.0, progress_s: float = 0.0) -> None:
        self._reference_time = float(now)
        self._reference_progress = float(progress_s)
        self._obstacle_since: Optional[float] = None
        self._blocked = False

    def update(
        self,
        *,
        now: float,
        valid_path: bool,
        distance_to_goal: float,
        linear_speed: float,
        angular_speed: float,
        progress_s: float,
        commanded_speed: float = 0.0,
        controller_valid: bool = True,
        path_obstructed: bool = False,
    ) -> BlockageObservation:
        cfg = self.config
        now = float(now)
        progress_s = float(progress_s)
        progress_delta = max(0.0, progress_s - self._reference_progress)

        if (not valid_path or distance_to_goal <= cfg.goal_tolerance):
            self.reset(now, progress_s)
            return BlockageObservation(False, 0.0, 0.0, "inactive_or_near_goal")

        if progress_delta >= cfg.blocked_progress_threshold:
            self._reference_time = now
            self._reference_progress = progress_s
            progress_delta = 0.0
            self._blocked = False

        if path_obstructed:
            if self._obstacle_since is None:
                self._obstacle_since = now
        else:
            self._obstacle_since = None

        stalled_for = max(0.0, now - self._reference_time)
        low_motion = (
            abs(linear_speed) < cfg.blocked_velocity_threshold
            and abs(angular_speed) < cfg.blocked_angular_threshold
        )
        commanding_without_motion = (
            abs(commanded_speed) >= cfg.blocked_velocity_threshold and low_motion)
        controller_failure = not controller_valid
        obstacle_persistent = (
            self._obstacle_since is not None
            and now - self._obstacle_since >= cfg.obstacle_confirmation_time)

        evidence = []
        if low_motion:
            evidence.append("low_motion")
        if commanding_without_motion:
            evidence.append("commanded_no_motion")
        if controller_failure:
            evidence.append("controller_invalid")
        if obstacle_persistent:
            evidence.append("persistent_obstacle")

        # A normal turn may have little path-index progress but meaningful yaw
        # motion.  Require low translational *and* angular motion, plus one
        # independent cause, before declaring a blockage.
        confirmed = (
            stalled_for >= cfg.blocked_timeout
            and low_motion
            and (commanding_without_motion or controller_failure or obstacle_persistent)
        )
        self._blocked = confirmed
        reason = "+".join(evidence) if evidence else "temporary_slowdown"
        return BlockageObservation(confirmed, stalled_for, progress_delta, reason)


class PathManager:
    """Cache a complete path and maintain constrained rolling execution state."""

    def __init__(self, config: Optional[PathManagerConfig] = None):
        self.config = config or PathManagerConfig()
        self.config.validate()
        self.complete_path: List[Point2] = []
        self.active_execution_path: List[Point2] = []
        self.detour_path: List[Point2] = []
        self.cumulative_s: List[float] = []
        self.hard_stop_indices: List[int] = []
        self.current_progress_index = 0
        self.current_progress_s = 0.0
        self.window_start_index = 0
        self.window_end_index = 0
        self.completed_path_range: List[PathRange] = []
        self.blocked_range: List[PathRange] = []
        self.skipped_range: List[PathRange] = []
        self.pending_recovery_areas: List[PathRange] = []
        self.path_version = 0
        self.path_version_reason = ""
        self.execution_state = ExecutionState.IDLE
        self.projected_point: Optional[Point2] = None
        self._dynamic_window_length = self.config.window_length
        self._pending_jump_s: Optional[float] = None
        self._pending_jump_frames = 0
        self._last_projection: Optional[PathProjection] = None
        self._last_window_progress_s = -math.inf
        self._last_window_end_s = -math.inf
        self._last_window_version = -1
        self._last_window_time = -math.inf
        self._last_adaptation_time: Optional[float] = None
        self.blockage_detector = BlockageDetector(self.config)

    @property
    def total_length(self) -> float:
        return self.cumulative_s[-1] if self.cumulative_s else 0.0

    @property
    def remaining_length(self) -> float:
        return max(0.0, self.total_length - self.current_progress_s)

    @staticmethod
    def _normalise_points(points: Iterable[Sequence[float]]) -> List[Point2]:
        out: List[Point2] = []
        for point in points:
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("path contains a non-finite point")
            if not out or math.hypot(x - out[-1][0], y - out[-1][1]) > 1e-7:
                out.append((x, y))
        return out

    @staticmethod
    def _normalise_points_with_index_map(
        points: Iterable[Sequence[float]],
    ) -> Tuple[List[Point2], List[int]]:
        """Remove zero-length edges and map each input index to its output index."""
        out: List[Point2] = []
        index_map: List[int] = []
        for point in points:
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("path contains a non-finite point")
            if not out or math.hypot(x - out[-1][0], y - out[-1][1]) > 1e-7:
                out.append((x, y))
            index_map.append(len(out) - 1)
        return out, index_map

    def set_complete_path(
        self,
        points: Iterable[Sequence[float]],
        *,
        hard_stop_indices: Sequence[int] = (),
        reason: str = "new_coverage_plan",
    ) -> int:
        path, source_to_path = self._normalise_points_with_index_map(points)
        if len(path) < 2:
            raise ValueError("complete coverage path requires at least two distinct points")
        cumulative = [0.0]
        for index in range(1, len(path)):
            cumulative.append(
                cumulative[-1]
                + math.hypot(path[index][0] - path[index - 1][0],
                             path[index][1] - path[index - 1][1]))
        if cumulative[-1] <= 0.0:
            raise ValueError("complete coverage path has zero length")

        self.complete_path = path
        self.active_execution_path = list(path)
        self.detour_path = []
        self.cumulative_s = cumulative
        self.hard_stop_indices = sorted({
            source_to_path[min(len(source_to_path) - 1, max(0, int(index)))]
            for index in hard_stop_indices
            if source_to_path and 0 < int(index) < len(source_to_path)
        })
        self.current_progress_index = 0
        self.current_progress_s = 0.0
        self.window_start_index = 0
        self.window_end_index = 0
        self.completed_path_range = []
        self.blocked_range = []
        self.skipped_range = []
        self.pending_recovery_areas = []
        self.path_version += 1
        self.path_version_reason = reason
        self.execution_state = ExecutionState.IDLE
        self.projected_point = path[0]
        self._pending_jump_s = None
        self._pending_jump_frames = 0
        self._last_projection = None
        self._last_window_progress_s = -math.inf
        self._last_window_end_s = -math.inf
        self._last_window_version = -1
        self._last_window_time = -math.inf
        self._dynamic_window_length = self.config.window_length
        self.blockage_detector.reset()
        return self.path_version

    def start_tracking(self, now: float = 0.0) -> None:
        if not self.complete_path:
            raise RuntimeError("cannot track without a complete path")
        self.execution_state = ExecutionState.TRACKING
        self.blockage_detector.reset(now, self.current_progress_s)

    def set_state(self, state: ExecutionState) -> None:
        self.execution_state = ExecutionState(state)

    def _segment_id_for_edge(self, edge_index: int) -> int:
        return bisect.bisect_right(self.hard_stop_indices, int(edge_index))

    def _segment_id_for_progress(self) -> int:
        edge = min(max(0, self.current_progress_index), max(0, len(self.complete_path) - 2))
        return self._segment_id_for_edge(edge)

    def _next_hard_stop_s(self, segment_id: int) -> Optional[float]:
        if segment_id < len(self.hard_stop_indices):
            return self.cumulative_s[self.hard_stop_indices[segment_id]]
        return None

    @staticmethod
    def _angle_error(a: float, b: float) -> float:
        return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)

    def _candidate_edges(self) -> range:
        low_s = max(0.0, self.current_progress_s - self.config.progress_search_backward_distance)
        high_s = min(
            self.total_length,
            self.current_progress_s + self.config.progress_search_forward_distance,
        )
        first = max(0, bisect.bisect_left(self.cumulative_s, low_s) - 1)
        last = min(len(self.complete_path) - 2,
                   bisect.bisect_right(self.cumulative_s, high_s))
        return range(first, last + 1)

    def _projection_candidates(
        self,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        enforce_heading: bool,
    ) -> List[Tuple[float, int, float, float, float, float, float]]:
        candidates = []
        current_segment = self._segment_id_for_progress()
        next_stop_s = self._next_hard_stop_s(current_segment)
        may_enter_next = (
            next_stop_s is None
            or self.current_progress_s >= next_stop_s - self.config.segment_transition_margin
        )
        for edge in self._candidate_edges():
            segment_id = self._segment_id_for_edge(edge)
            if segment_id < max(0, current_segment - 1):
                continue
            if segment_id > current_segment + (1 if may_enter_next else 0):
                continue
            ax, ay = self.complete_path[edge]
            bx, by = self.complete_path[edge + 1]
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            if length2 <= 1e-12:
                continue
            t = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length2))
            px, py = ax + t * dx, ay + t * dy
            distance = math.hypot(x - px, y - py)
            if distance > self.config.max_cross_track_error:
                continue
            tangent = math.atan2(dy, dx)
            heading_error = self._angle_error(yaw, tangent)
            # While rotating in place, heading is not reliable evidence for
            # which lane the chassis belongs to.  At normal speed it is a hard
            # anti-jump gate.
            if (enforce_heading and abs(speed) >= self.config.blocked_velocity_threshold
                    and heading_error > self.config.max_heading_error):
                continue
            seg_len = math.sqrt(length2)
            s = self.cumulative_s[edge] + t * seg_len
            delta = s - self.current_progress_s
            rollback_penalty = max(0.0, -delta) * 5.0
            advance_penalty = (max(0.0, delta) / max(
                self.config.progress_search_forward_distance, 1e-6)) ** 2
            score = (distance * distance * 6.0
                     + heading_error * heading_error * (0.8 if enforce_heading else 0.1)
                     + rollback_penalty + advance_penalty)
            candidates.append((score, edge, s, px, py, distance, heading_error))
        return candidates

    def update_progress(
        self,
        pose: Sequence[float],
        *,
        speed: float = 0.0,
    ) -> Optional[PathProjection]:
        if len(self.complete_path) < 2:
            return None
        x, y = float(pose[0]), float(pose[1])
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        candidates = self._projection_candidates(x, y, yaw, speed, True)
        if (not candidates
                and abs(speed) < self.config.blocked_velocity_threshold):
            candidates = self._projection_candidates(x, y, yaw, speed, False)
        if not candidates:
            return None
        _, edge, raw_s, px, py, cross_track, heading_error = min(candidates)

        confirmed = True
        advance = raw_s - self.current_progress_s
        if advance > self.config.progress_jump_confirm_distance:
            if (self._pending_jump_s is not None
                    and abs(raw_s - self._pending_jump_s)
                    <= self.config.progress_jump_confirm_distance * 0.5):
                self._pending_jump_frames += 1
            else:
                self._pending_jump_s = raw_s
                self._pending_jump_frames = 1
            confirmed = self._pending_jump_frames >= self.config.progress_jump_confirm_frames
        else:
            self._pending_jump_s = None
            self._pending_jump_frames = 0

        if confirmed:
            # Committed task progress is monotonic.  Small backwards raw
            # projections remain visible in diagnostics but never resurrect a
            # completed lane.
            self.current_progress_s = max(self.current_progress_s, raw_s)
            self.current_progress_index = min(
                len(self.complete_path) - 1,
                max(self.current_progress_index,
                    bisect.bisect_right(self.cumulative_s, self.current_progress_s) - 1),
            )
            self.projected_point = (px, py)

        projection = PathProjection(
            index=self.current_progress_index,
            segment_index=edge,
            segment_id=self._segment_id_for_edge(edge),
            s=raw_s,
            x=px,
            y=py,
            cross_track_error=cross_track,
            heading_error=heading_error,
            confirmed=confirmed,
        )
        self._last_projection = projection
        return projection

    def _curvature_ahead(self, distance: float = 10.0) -> float:
        if len(self.complete_path) < 3:
            return 0.0
        start = min(len(self.complete_path) - 2, self.current_progress_index)
        limit_s = min(self.total_length, self.current_progress_s + max(1.0, distance))
        previous_heading: Optional[float] = None
        turn = 0.0
        travelled = 0.0
        for edge in range(start, len(self.complete_path) - 1):
            if self.cumulative_s[edge] > limit_s:
                break
            a, b = self.complete_path[edge], self.complete_path[edge + 1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            step = math.hypot(dx, dy)
            if step <= 1e-7:
                continue
            heading = math.atan2(dy, dx)
            if previous_heading is not None:
                turn += self._angle_error(heading, previous_heading)
            previous_heading = heading
            travelled += step
        return turn / max(travelled, 1e-6)

    def adapt_window_length(
        self,
        *,
        now: float,
        speed: float = 0.0,
        obstacle_density: float = 0.0,
        narrow_area: bool = False,
    ) -> float:
        cfg = self.config
        target = cfg.window_length + max(0.0, abs(speed) - 0.20) * 20.0
        curvature = self._curvature_ahead()
        if curvature > 0.45:
            target = min(target, 22.0)
        elif curvature > 0.20:
            target = min(target, 30.0)
        elif curvature < 0.04 and obstacle_density < 0.03:
            target = max(target, 50.0)
        if narrow_area:
            target = min(target, 25.0)
        density = min(1.0, max(0.0, float(obstacle_density)))
        target -= 20.0 * density
        target = min(cfg.max_window_length, max(cfg.min_window_length, target))

        if self._last_adaptation_time is None:
            self._last_adaptation_time = float(now)
            self._dynamic_window_length = target
            return target
        dt = max(0.0, float(now) - self._last_adaptation_time)
        self._last_adaptation_time = float(now)
        max_change = cfg.window_length_change_rate * dt
        delta = min(max_change, max(-max_change, target - self._dynamic_window_length))
        self._dynamic_window_length += delta
        return self._dynamic_window_length

    def _point_at_s(self, s: float) -> Tuple[float, float, float, int]:
        s = min(self.total_length, max(0.0, float(s)))
        edge = min(len(self.complete_path) - 2,
                   max(0, bisect.bisect_right(self.cumulative_s, s) - 1))
        a, b = self.complete_path[edge], self.complete_path[edge + 1]
        seg_len = max(1e-12, self.cumulative_s[edge + 1] - self.cumulative_s[edge])
        t = min(1.0, max(0.0, (s - self.cumulative_s[edge]) / seg_len))
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        return x, y, yaw, edge

    def _slice_between(self, start_s: float, end_s: float) -> Tuple[ManagedPose, ...]:
        start_s = min(self.total_length, max(0.0, float(start_s)))
        end_s = min(self.total_length, max(start_s, float(end_s)))
        sx, sy, syaw, _ = self._point_at_s(start_s)
        out = [ManagedPose(sx, sy, syaw, start_s)]
        first = bisect.bisect_right(self.cumulative_s, start_s)
        last = bisect.bisect_left(self.cumulative_s, end_s)
        for index in range(first, last):
            x, y = self.complete_path[index]
            if index < len(self.complete_path) - 1:
                nx, ny = self.complete_path[index + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                px, py = self.complete_path[index - 1]
                yaw = math.atan2(y - py, x - px)
            if math.hypot(x - out[-1].x, y - out[-1].y) > 1e-7:
                out.append(ManagedPose(x, y, yaw, self.cumulative_s[index]))
        ex, ey, eyaw, _ = self._point_at_s(end_s)
        if math.hypot(ex - out[-1].x, ey - out[-1].y) > 1e-7:
            out.append(ManagedPose(ex, ey, eyaw, end_s))
        elif out:
            out[-1] = ManagedPose(out[-1].x, out[-1].y, eyaw, end_s)
        return tuple(out)

    def slice_between(self, start_s: float, end_s: float) -> Tuple[ManagedPose, ...]:
        """Public, bounds-checked arc-length slice for debug and recovery users."""
        if len(self.complete_path) < 2:
            return ()
        return self._slice_between(start_s, end_s)

    def build_window(self, lookahead_distance: Optional[float] = None) -> PathWindow:
        if len(self.complete_path) < 2:
            raise RuntimeError("cannot build a window without a complete path")
        lookahead = self._dynamic_window_length if lookahead_distance is None else float(lookahead_distance)
        lookahead = min(self.config.max_window_length,
                        max(self.config.min_window_length, lookahead))
        start_s = max(0.0, self.current_progress_s - self.config.backtrack_margin)
        end_s = min(self.total_length, self.current_progress_s + lookahead)
        end_s = self._safe_window_endpoint(end_s)
        poses = self._slice_between(start_s, end_s)
        start_index = max(0, bisect.bisect_right(self.cumulative_s, start_s) - 1)
        end_index = min(len(self.complete_path) - 1,
                        bisect.bisect_left(self.cumulative_s, end_s))
        self.window_start_index = start_index
        self.window_end_index = end_index
        return PathWindow(
            poses=poses,
            start_s=start_s,
            end_s=end_s,
            start_index=start_index,
            end_index=end_index,
            path_version=self.path_version,
            remaining_length=self.remaining_length,
        )

    def _safe_window_endpoint(self, desired_end_s: float) -> float:
        """Avoid a long folded window whose action goal lies beside the robot.

        FollowPath's goal checker is geometric.  On a dense snake/closed loop,
        a point tens of metres ahead in arc length can still be beside the
        current pose and cause an immediate false success.  Keep the requested
        lookahead when possible; otherwise pick the closest safe arc-length
        endpoint within the configured window bounds.
        """
        desired = min(self.total_length, max(self.current_progress_s, desired_end_s))
        if desired - self.current_progress_s <= max(1.0, self.config.goal_tolerance * 4.0):
            return desired
        cx, cy, _, _ = self._point_at_s(self.current_progress_s)

        def safe(candidate_s: float) -> bool:
            x, y, _, _ = self._point_at_s(candidate_s)
            return math.hypot(x - cx, y - cy) >= self.config.min_goal_endpoint_distance

        if safe(desired):
            return desired
        low = min(
            self.total_length,
            self.current_progress_s + self.config.min_window_length,
        )
        high = min(
            self.total_length,
            self.current_progress_s + self.config.max_window_length,
        )
        step = max(0.25, self.config.path_point_spacing * 2.0)
        count = int(math.ceil(max(desired - low, high - desired) / step))
        for offset_index in range(1, count + 1):
            offset = offset_index * step
            for candidate in (desired - offset, desired + offset):
                if candidate < low - 1e-9 or candidate > high + 1e-9:
                    continue
                if safe(candidate):
                    return candidate
        return desired

    def effective_mppi_reference_length(self, speed: float = 0.0) -> float:
        cfg = self.config
        required = abs(float(speed)) * cfg.mppi_prediction_horizon + cfg.mppi_safety_margin
        desired = max(cfg.mppi_reference_min_length,
                      cfg.mppi_reference_length, required)
        desired = min(cfg.mppi_reference_max_length, desired)
        costmap_cap = max(0.5, cfg.mppi_costmap_radius - cfg.mppi_costmap_margin)
        return min(desired, costmap_cap, self.remaining_length + cfg.backtrack_margin)

    def build_mppi_reference(self, speed: float = 0.0) -> PathWindow:
        length = self.effective_mppi_reference_length(speed)
        start_s = max(0.0, self.current_progress_s - min(0.25, self.config.backtrack_margin))
        end_s = min(self.total_length, self.current_progress_s + max(0.0, length))
        poses = self._slice_between(start_s, end_s)
        return PathWindow(
            poses=poses,
            start_s=start_s,
            end_s=end_s,
            start_index=max(0, bisect.bisect_right(self.cumulative_s, start_s) - 1),
            end_index=min(len(self.complete_path) - 1,
                          bisect.bisect_left(self.cumulative_s, end_s)),
            path_version=self.path_version,
            remaining_length=self.remaining_length,
        )

    def poses_for_ranges(self, ranges: Sequence[PathRange]) -> Tuple[ManagedPose, ...]:
        """Return ordered debug poses for path ranges.

        Range boundaries are sampled by arc length.  A repeated boundary pose
        separates disjoint ranges for consumers that split on zero-length
        pairs; RViz Path still provides a compact best-effort overview while
        ``/coverage/debug_markers`` renders each range independently.
        """
        out: List[ManagedPose] = []
        for item in sorted(ranges, key=lambda value: value.start_s):
            part = list(self._slice_between(item.start_s, item.end_s))
            if not part:
                continue
            if out and math.hypot(part[0].x - out[-1].x, part[0].y - out[-1].y) > 1e-7:
                out.append(out[-1])
            out.extend(part)
        return tuple(out)

    def needs_window_update(self, now: float, window: Optional[PathWindow] = None) -> bool:
        if self.execution_state != ExecutionState.TRACKING:
            return False
        window = window or self.build_window()
        if self._last_window_version != self.path_version:
            return True
        if self.current_progress_s - self._last_window_progress_s >= self.config.path_update_distance:
            return True
        ahead_remaining = self._last_window_end_s - self.current_progress_s
        if ahead_remaining <= self.config.min_window_remaining_length:
            return True
        min_period = 1.0 / self.config.path_update_rate
        content_changed = abs(window.end_s - self._last_window_end_s) >= self.config.path_point_spacing
        return content_changed and float(now) - self._last_window_time >= min_period

    def mark_window_published(self, window: PathWindow, now: float) -> None:
        self._last_window_progress_s = self.current_progress_s
        self._last_window_end_s = window.end_s
        self._last_window_version = window.path_version
        self._last_window_time = float(now)

    @staticmethod
    def _merge_ranges(ranges: List[PathRange], new_range: PathRange) -> List[PathRange]:
        ordered = sorted(ranges + [new_range], key=lambda item: (item.start_s, item.end_s))
        merged: List[PathRange] = []
        for item in ordered:
            lo, hi = min(item.start_s, item.end_s), max(item.start_s, item.end_s)
            if not merged or lo > merged[-1].end_s + 1e-6 or item.reason != merged[-1].reason:
                merged.append(PathRange(lo, hi, item.reason))
            else:
                previous = merged[-1]
                merged[-1] = PathRange(previous.start_s, max(previous.end_s, hi), previous.reason)
        return merged

    def mark_completed(self, start_s: float, end_s: float, reason: str = "executed") -> None:
        item = PathRange(max(0.0, start_s), min(self.total_length, end_s), reason)
        self.completed_path_range = self._merge_ranges(self.completed_path_range, item)

    def mark_blocked(self, start_s: float, end_s: float, reason: str) -> None:
        item = PathRange(max(0.0, start_s), min(self.total_length, end_s), reason)
        self.blocked_range = self._merge_ranges(self.blocked_range, item)
        self.execution_state = ExecutionState.BLOCKED

    def mark_skipped(self, start_s: float, end_s: float, reason: str) -> None:
        lo, hi = max(0.0, start_s), min(self.total_length, end_s)
        if hi - lo > self.config.skipped_segment_max_length + 1e-6:
            raise ValueError("skipped range exceeds skipped_segment_max_length")
        item = PathRange(lo, hi, reason)
        self.skipped_range = self._merge_ranges(self.skipped_range, item)
        self.pending_recovery_areas = self._merge_ranges(self.pending_recovery_areas, item)

    def set_detour(self, points: Iterable[Sequence[float]], reason: str = "local_reconnect") -> None:
        self.detour_path = self._normalise_points(points)
        self.active_execution_path = list(self.detour_path)
        self.path_version += 1
        self.path_version_reason = reason
        self.execution_state = ExecutionState.REPLANNING

    def resume_complete_path(self, reason: str = "rejoined_complete_path") -> None:
        """Return execution ownership to the immutable coverage route."""
        self.active_execution_path = list(self.complete_path)
        self.detour_path = []
        self.path_version += 1
        self.path_version_reason = reason
        self.execution_state = ExecutionState.TRACKING

    def completion_ready(
        self,
        *,
        coverage_ratio: float,
        target_coverage_ratio: float = 0.95,
        remaining_path_threshold: float = 0.5,
        reachable_pending_areas: int = 0,
    ) -> bool:
        return (
            self.remaining_length <= remaining_path_threshold
            and float(coverage_ratio) >= float(target_coverage_ratio)
            and int(reachable_pending_areas) == 0
            and self.execution_state not in {
                ExecutionState.BLOCKED,
                ExecutionState.RECOVERING,
                ExecutionState.REPLANNING,
                ExecutionState.FAILED,
            }
        )

    def progress_message(self) -> dict:
        return {
            "path_version": self.path_version,
            "path_version_reason": self.path_version_reason,
            "state": self.execution_state.value,
            "current_progress_index": self.current_progress_index,
            "current_progress_s": self.current_progress_s,
            "total_length": self.total_length,
            "remaining_length": self.remaining_length,
            "window_start_index": self.window_start_index,
            "window_end_index": self.window_end_index,
            "window_length": self._dynamic_window_length,
            "completed_ranges": [asdict(item) for item in self.completed_path_range],
            "blocked_ranges": [asdict(item) for item in self.blocked_range],
            "skipped_ranges": [asdict(item) for item in self.skipped_range],
            "pending_recovery_areas": [asdict(item) for item in self.pending_recovery_areas],
        }
