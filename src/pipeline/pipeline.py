"""
AdaptiveLidarPipeline (Module 38) — the single integration point.
`result = pipeline.process(point_cloud)` runs every stage and returns a
PipelineResult carrying each stage's output, plus per-stage timing (Module 39
logging) so the dashboard/benchmark can report latency breakdowns, not just
a black-box total.

For sequential data, `process_stream` feeds successive frames through the
same tracker + temporal map manager so dynamic objects are tracked and
stale map entries decay, matching spec section 25.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from src.common.config import Config
from src.common.pointcloud import PointCloud
from src.mapping.adaptive_grid import AdaptiveGrid, build_resolution_function
from src.mapping.projection import project_to_grid
from src.mapping.temporal import TemporalMapManager
from src.objects.clustering import DetectedObject, classify_by_size, cluster_obstacles
from src.objects.tracker import ObjectTracker
from src.preprocessing.filtering import range_filter, statistical_outlier_removal, voxel_downsample
from src.segmentation.base import SegmentationModel, SegmentationResult
from src.segmentation.heuristic import HeuristicSegmenter, TERRAIN, STATIC_OBSTACLE
from src.terrain.terrain_analyzer import TerrainResult, analyze_terrain

logger = logging.getLogger("adaptive_lidar_pipeline")


@dataclass
class PipelineResult:
    frame_timestamp: float
    n_points_raw: int
    n_points_processed: int
    processed_pointcloud: PointCloud
    semantic: SegmentationResult
    terrain: TerrainResult
    objects: List[DetectedObject]
    adaptive_grid: AdaptiveGrid
    metrics: Dict[str, float] = field(default_factory=dict)
    model_mode: str = "baseline"


class AdaptiveLidarPipeline:
    def __init__(self, config: Config, segmenter: Optional[SegmentationModel] = None):
        self.config = config
        self.segmenter = segmenter or HeuristicSegmenter(
            ground_z_percentile=config.terrain.ground_z_percentile,
            neighbor_radius=config.terrain.neighbor_radius,
            obstacle_height_threshold=0.15,
        )
        self.tracker = ObjectTracker(
            max_match_distance=config.objects.tracking.max_match_distance,
            dynamic_speed_threshold=config.objects.tracking.dynamic_speed_threshold,
            max_missed_frames=config.objects.tracking.max_missed_frames,
        )
        self.temporal_manager = TemporalMapManager(
            decay=config.mapping.temporal_decay,
            dynamic_timeout_frames=config.mapping.dynamic_object_timeout_frames,
        )
        self.grid = AdaptiveGrid(
            build_resolution_function(config.grid),
            max_range=config.sensor.max_range,
        )

    def process(self, pc: PointCloud) -> PipelineResult:
        metrics: Dict[str, float] = {}
        n_raw = len(pc)
        t_start = time.perf_counter()

        # --- Preprocessing ---
        t0 = time.perf_counter()
        try:
            pc_f = range_filter(pc, self.config.sensor.min_range, self.config.sensor.max_range)
            if self.config.preprocessing.statistical_outlier.enabled and len(pc_f) > 0:
                pc_f = statistical_outlier_removal(
                    pc_f,
                    k_neighbors=self.config.preprocessing.statistical_outlier.k_neighbors,
                    std_ratio=self.config.preprocessing.statistical_outlier.std_ratio,
                )
            if self.config.preprocessing.voxel_downsample.enabled and len(pc_f) > 0:
                pc_f = voxel_downsample(pc_f, self.config.preprocessing.voxel_downsample.voxel_size)
        except Exception:
            logger.exception("Preprocessing failed for frame at t=%s; skipping frame", pc.frame_timestamp)
            pc_f = pc.subset(np.zeros(len(pc), dtype=bool))  # empty -> pipeline degrades gracefully
        metrics["preprocessing_ms"] = (time.perf_counter() - t0) * 1000.0

        n_processed = len(pc_f)
        if n_processed == 0:
            logger.warning("Frame at t=%s has no points after preprocessing", pc.frame_timestamp)

        # --- Segmentation ---
        t0 = time.perf_counter()
        try:
            seg = self.segmenter.predict(pc_f)
        except Exception:
            logger.exception("Segmentation failed; falling back to all-unknown labels")
            seg = SegmentationResult(
                semantic_class=np.zeros(n_processed, dtype=np.int32),
                confidence=np.zeros(n_processed, dtype=np.float32),
                mode="baseline",
            )
        metrics["segmentation_ms"] = (time.perf_counter() - t0) * 1000.0

        # --- Terrain analysis ---
        t0 = time.perf_counter()
        try:
            terrain = analyze_terrain(
                pc_f, seg,
                neighbor_radius=self.config.terrain.neighbor_radius,
                max_slope_deg=self.config.terrain.max_drivable_slope_deg,
                max_height_diff=self.config.terrain.max_drivable_height_diff,
                roughness_threshold=self.config.terrain.roughness_threshold,
            )
        except Exception:
            logger.exception("Terrain analysis failed; using neutral terrain result")
            empty = np.zeros(n_processed, dtype=np.float32)
            terrain = TerrainResult(empty, empty, np.zeros(n_processed, dtype=bool), empty)
        metrics["terrain_ms"] = (time.perf_counter() - t0) * 1000.0

        # --- Object detection + tracking ---
        t0 = time.perf_counter()
        obstacle_mask = np.isin(seg.semantic_class, [STATIC_OBSTACLE])
        try:
            detections = cluster_obstacles(
                pc_f, obstacle_mask,
                eps=self.config.objects.clustering.eps,
                min_samples=self.config.objects.clustering.min_samples,
            )
            for obj in detections:
                obj.meta_class_hint = classify_by_size(  # type: ignore[attr-defined]
                    obj,
                    pedestrian_max_footprint=self.config.objects.size_heuristics.pedestrian_max_footprint,
                    pedestrian_max_height=self.config.objects.size_heuristics.pedestrian_max_height,
                    vehicle_min_footprint=self.config.objects.size_heuristics.vehicle_min_footprint,
                )
            detections = self.tracker.update(detections, timestamp=pc.frame_timestamp)
        except Exception:
            logger.exception("Object detection/tracking failed; continuing with no objects this frame")
            detections = []
        metrics["objects_ms"] = (time.perf_counter() - t0) * 1000.0

        # --- Adaptive projection + map fusion ---
        t0 = time.perf_counter()
        observed_keys = set()
        try:
            if n_processed > 0:
                project_to_grid(pc_f, seg, terrain, self.grid, timestamp=pc.frame_timestamp, objects=detections)
                for i in range(n_processed):
                    key = self.grid.get_cell_key(float(pc_f.points[i, 0]), float(pc_f.points[i, 1]))
                    if key is not None:
                        observed_keys.add(key)
            self.temporal_manager.update(self.grid, observed_keys)
        except Exception:
            logger.exception("Projection/map update failed for this frame")
        metrics["projection_ms"] = (time.perf_counter() - t0) * 1000.0

        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        metrics["fps"] = 1000.0 / metrics["total_ms"] if metrics["total_ms"] > 0 else float("inf")
        metrics["cell_count"] = float(self.grid.cell_count())
        metrics["point_count"] = float(n_processed)

        logger.info(
            "frame t=%.3f points_raw=%d points_processed=%d total_ms=%.2f fps=%.1f cells=%d",
            pc.frame_timestamp, n_raw, n_processed, metrics["total_ms"], metrics["fps"], self.grid.cell_count(),
        )

        return PipelineResult(
            frame_timestamp=pc.frame_timestamp,
            n_points_raw=n_raw,
            n_points_processed=n_processed,
            processed_pointcloud=pc_f,
            semantic=seg,
            terrain=terrain,
            objects=detections,
            adaptive_grid=self.grid,
            metrics=metrics,
            model_mode=seg.mode,
        )

    def process_stream(self, frame_generator: Iterable[PointCloud]) -> Iterable[PipelineResult]:
        for pc in frame_generator:
            yield self.process(pc)
