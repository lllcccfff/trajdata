import os
from collections import deque
from pathlib import Path
from subprocess import check_call, check_output
from typing import Dict, Final, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import map_pb2 as waymo_map_pb2
from waymo_open_dataset.protos import scenario_pb2

from trajdata.maps import TrafficLightStatus, VectorMap
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    Polyline,
    Road,
    RoadArea,
    RoadAreaPart,
    RoadLane,
)

WAYMO_DT: Final[float] = 0.1
WAYMO_DATASET_NAMES = [
    "testing",
    "testing_interactive",
    "training",
    "training_20s",
    "validation",
    "validation_interactive",
]

TRAIN_SCENE_LENGTH = 91
VAL_SCENE_LENGTH = 91
TEST_SCENE_LENGTH = 11
TRAIN_20S_SCENE_LENGTH = 201

GREEN = [
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_GO,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_GO,
]
RED = [
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_CAUTION,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_CAUTION,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_STOP,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_CAUTION,
]

from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)


class WaymoScenarios:
    def __init__(
        self,
        dataset_name: str,
        source_dir: Path,
        download: bool = False,
        split: bool = False,
    ):
        if dataset_name not in WAYMO_DATASET_NAMES:
            raise RuntimeError(
                "Wrong dataset name. Please choose name from "
                + str(WAYMO_DATASET_NAMES)
            )

        self.name = dataset_name
        self.source_dir = source_dir
        if dataset_name in ["training"]:
            self.scene_length = TRAIN_SCENE_LENGTH
        elif dataset_name in ["validation", "validation_interactive"]:
            self.scene_length = VAL_SCENE_LENGTH
        elif dataset_name in ["testing", "testing_interactive"]:
            self.scene_length = TEST_SCENE_LENGTH
        elif dataset_name in ["training_20s"]:
            self.scene_length = TRAIN_20S_SCENE_LENGTH

        if download:
            self.download_dataset()

        split_path = self.source_dir / (self.name + "_splitted")
        if split or not split_path.is_dir():
            self.split_scenarios()
        else:
            self.num_scenarios = len(os.listdir(split_path))

    def download_dataset(self) -> None:
        # check_call("snap install google-cloud-sdk --classic".split())
        gsutil = check_output(["which", "gsutil"])
        download_cmd = (
            str(gsutil.decode("utf-8"))
            + "-m cp -r gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/"
            + str(self.name)
            + " "
            + str(self.source_dir)
        ).split()
        check_call(download_cmd)

    def split_scenarios(
        self, num_parallel_reads: int = 20, verbose: bool = True
    ) -> None:
        source_it: Path = (self.source_dir / self.name).glob("*")
        file_names: List[str] = [str(file_name) for file_name in source_it]
        if verbose:
            print("Loading tfrecord files...")
        dataset = tf.data.TFRecordDataset(
            file_names, compression_type="", num_parallel_reads=num_parallel_reads
        )

        if verbose:
            print("Splitting tfrecords...")

        splitted_dir: Path = self.source_dir / f"{self.name}_splitted"
        if not splitted_dir.exists():
            splitted_dir.mkdir(parents=True)

        scenario_num: int = 0
        for data in tqdm(dataset):
            file_name: Path = (
                splitted_dir / f"{self.name}_splitted_{scenario_num}.tfrecords"
            )
            with tf.io.TFRecordWriter(str(file_name)) as file_writer:
                file_writer.write(data.numpy())

            scenario_num += 1

        self.num_scenarios = scenario_num
        if verbose:
            print(
                str(self.num_scenarios)
                + " scenarios from "
                + str(len(file_names))
                + " file(s) have been split into "
                + str(self.num_scenarios)
                + " files."
            )

    def get_filename(self, data_idx):
        return (
            self.source_dir
            / f"{self.name}_splitted"
            / f"{self.name}_splitted_{data_idx}.tfrecords"
        )


def extract_vectorized(
    map_features: List[waymo_map_pb2.MapFeature], map_name: str, verbose: bool = False
) -> VectorMap:
    vec_map = VectorMap(map_id=map_name)

    max_pt = np.array([np.nan, np.nan, np.nan])
    min_pt = np.array([np.nan, np.nan, np.nan])

    boundaries: Dict[int, Polyline] = {}
    for map_feature in tqdm(
        map_features, desc="Extracting road boundaries", disable=not verbose
    ):
        if map_feature.WhichOneof("feature_data") == "road_line":
            boundaries[map_feature.id] = Polyline(
                np.array([(pt.x, pt.y, pt.z) for pt in map_feature.road_line.polyline])
            )
        elif map_feature.WhichOneof("feature_data") == "road_edge":
            boundaries[map_feature.id] = Polyline(
                np.array([(pt.x, pt.y, pt.z) for pt in map_feature.road_edge.polyline])
            )

    for map_feature in tqdm(
        map_features, desc="Extracting map elements", disable=not verbose
    ):
        if map_feature.WhichOneof("feature_data") == "lane":
            if len(map_feature.lane.polyline) == 1:
                # TODO: Why does Waymo have single-point polylines that
                # aren't interpolating between others??
                continue

            road_lane = translate_lane(map_feature, boundaries)
            vec_map.add_map_element(road_lane)

            max_pt = np.fmax(max_pt, road_lane.center.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, road_lane.center.xyz.min(axis=0))

            if road_lane.left_edge:
                max_pt = np.fmax(max_pt, road_lane.left_edge.xyz.max(axis=0))
                min_pt = np.fmin(min_pt, road_lane.left_edge.xyz.min(axis=0))

            if road_lane.right_edge:
                max_pt = np.fmax(max_pt, road_lane.right_edge.xyz.max(axis=0))
                min_pt = np.fmin(min_pt, road_lane.right_edge.xyz.min(axis=0))

        elif map_feature.WhichOneof("feature_data") == "crosswalk":
            crosswalk = PedCrosswalk(
                id=str(map_feature.id),
                polygon=Polyline(
                    np.array(
                        [(pt.x, pt.y, pt.z) for pt in map_feature.crosswalk.polygon]
                    )
                ),
            )
            vec_map.add_map_element(crosswalk)

            max_pt = np.fmax(max_pt, crosswalk.polygon.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, crosswalk.polygon.xyz.min(axis=0))

        else:
            continue

    add_waymo_road_topology(vec_map)
    for road_area in vec_map.elements[MapElementType.ROAD_AREA].values():
        for road_area_part in road_area.areas:
            max_pt = np.fmax(max_pt, road_area_part.exterior_polygon.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, road_area_part.exterior_polygon.xyz.min(axis=0))

    # Setting the map bounds.
    # vec_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vec_map.extent = np.concatenate((min_pt, max_pt))

    return vec_map


def translate_agent_type(agent_type):
    if agent_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
        return AgentType.VEHICLE
    elif agent_type == scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
        return AgentType.PEDESTRIAN
    elif agent_type == scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
        return AgentType.BICYCLE
    elif agent_type == scenario_pb2.Track.ObjectType.OTHER:
        return AgentType.UNKNOWN
    return -1


def is_full_boundary(lane_boundaries, num_lane_indices: int) -> bool:
    """Returns True if a given boundary is connected (there are no gaps)
    and every lane center index has a corresponding boundary point.

    Returns:
        bool
    """
    covers_all: bool = lane_boundaries[0].lane_start_index == 0 and lane_boundaries[
        0
    ].lane_end_index == (num_lane_indices - 1)
    for idx in range(1, len(lane_boundaries)):
        if (
            lane_boundaries[idx].lane_start_index
            != lane_boundaries[idx - 1].lane_end_index + 1
        ):
            covers_all = False
            break

    return covers_all


def _centerline_polyline(lane: waymo_map_pb2.LaneCenter) -> Polyline:
    return Polyline(np.array([(pt.x, pt.y, pt.z) for pt in lane.polyline]))


def _aligned_segment_points(
    segment: np.ndarray, previous_end: Optional[np.ndarray]
) -> np.ndarray:
    if previous_end is None or segment.shape[0] < 2:
        return segment

    forward_dist = np.linalg.norm(segment[0, :2] - previous_end[:2])
    reverse_dist = np.linalg.norm(segment[-1, :2] - previous_end[:2])
    if reverse_dist < forward_dist:
        return segment[::-1]
    return segment


def _append_polyline_segment(points: List[np.ndarray], segment: np.ndarray) -> None:
    if len(points) == 0:
        points.extend(segment)
        return

    start_idx = 1 if np.allclose(points[-1], segment[0]) else 0
    points.extend(segment[start_idx:])


def _boundary_polyline(
    lane_boundaries,
    boundaries: Dict[int, Polyline],
) -> Optional[Polyline]:
    if len(lane_boundaries) == 0:
        return None

    points: List[np.ndarray] = []
    for boundary_ref in sorted(
        lane_boundaries, key=lambda item: (item.lane_start_index, item.lane_end_index)
    ):
        boundary = boundaries[boundary_ref.boundary_feature_id].xyz
        segment = _aligned_segment_points(boundary, points[-1] if points else None)
        _append_polyline_segment(points, segment)

    return Polyline(np.asarray(points))


def _edge_from_center(center: Polyline, lateral_offset_m: float) -> Polyline:
    center_xyz = center.xyz
    headings = center.h
    normals = np.stack((-np.sin(headings), np.cos(headings)), axis=1)
    shifted = center_xyz.copy()
    shifted[:, :2] = shifted[:, :2] + lateral_offset_m * normals
    return Polyline(shifted)


def _angle_diff_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _mean_nearby_heading_diff(lane: RoadLane, neighbor: RoadLane) -> float:
    lane_xy = lane.center.xy
    neighbor_xy = neighbor.center.xy
    dists = np.linalg.norm(lane_xy[:, np.newaxis, :] - neighbor_xy[np.newaxis, :, :], axis=2)
    neighbor_inds = np.argmin(dists, axis=1)
    return float(np.mean(_angle_diff_abs(lane.center.h, neighbor.center.h[neighbor_inds])))


def _filter_opposite_direction_adjacency(
    lanes: Dict[str, RoadLane], max_heading_diff: float = np.deg2rad(15.0)
) -> None:
    for lane in lanes.values():
        lane.adj_lanes_left = {
            lane_id
            for lane_id in lane.adj_lanes_left
            if lane_id in lanes
            and _mean_nearby_heading_diff(lane, lanes[lane_id]) < max_heading_diff
        }
        lane.adj_lanes_right = {
            lane_id
            for lane_id in lane.adj_lanes_right
            if lane_id in lanes
            and _mean_nearby_heading_diff(lane, lanes[lane_id]) < max_heading_diff
        }


def _lane_order_key(reference_lane: RoadLane, candidate_lane: RoadLane) -> float:
    projected = reference_lane.center.project_onto(candidate_lane.center.xyz)
    lateral = candidate_lane.center.xy - projected[:, :2]
    heading = projected[:, 3]
    left_normals = np.stack((-np.sin(heading), np.cos(heading)), axis=1)
    return float(np.mean(np.sum(lateral * left_normals, axis=1)))


def _ordered_road_lanes(lanes: Dict[str, RoadLane], lane_ids: Set[str]) -> List[RoadLane]:
    reference_lane = lanes[sorted(lane_ids)[0]]
    return sorted(
        (lanes[lane_id] for lane_id in lane_ids),
        key=lambda lane: _lane_order_key(reference_lane, lane),
        reverse=True,
    )


def _closed_road_polygon(lanes: Dict[str, RoadLane], lane_ids: Set[str]) -> Polyline:
    ordered_lanes = _ordered_road_lanes(lanes, lane_ids)
    left_edge = ordered_lanes[0].left_edge
    right_edge = ordered_lanes[-1].right_edge
    if left_edge is None or right_edge is None:
        raise ValueError("Road polygon requires lane edges.")

    points = np.concatenate((left_edge.xyz, right_edge.xyz[::-1]), axis=0)
    if not np.allclose(points[0], points[-1]):
        points = np.concatenate((points, points[[0]]), axis=0)
    return Polyline(points)


def add_waymo_road_topology(vec_map: VectorMap) -> None:
    lanes: Dict[str, RoadLane] = vec_map.elements[MapElementType.ROAD_LANE]
    _filter_opposite_direction_adjacency(lanes)

    visited: Set[str] = set()
    road_to_lanes: Dict[str, Set[str]] = {}
    lane_to_road: Dict[str, str] = {}

    for lane_id in sorted(lanes):
        if lane_id in visited:
            continue

        road_id = f"{vec_map.map_id}:road_{len(road_to_lanes)}"
        collected: Set[str] = set()
        queue = deque([lane_id])
        visited.add(lane_id)

        while queue:
            current_id = queue.popleft()
            collected.add(current_id)
            current_lane = lanes[current_id]
            for neighbor_id in sorted(current_lane.adj_lanes_left | current_lane.adj_lanes_right):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append(neighbor_id)

        road_to_lanes[road_id] = collected
        for road_lane_id in collected:
            lane_to_road[road_lane_id] = road_id

    vec_map.elements[MapElementType.ROAD].clear()
    vec_map.elements[MapElementType.ROAD_AREA].clear()
    vec_map.lane_to_road.clear()
    vec_map.roads = None

    for road_id, lane_ids in road_to_lanes.items():
        vec_map.add_map_element(
            RoadArea(
                id=road_id,
                areas=[RoadAreaPart(exterior_polygon=_closed_road_polygon(lanes, lane_ids))],
            )
        )
        vec_map.add_map_element(
            Road(id=road_id, road_area_id=road_id, lane_ids=lane_ids)
        )

    roads: Dict[str, Road] = vec_map.elements[MapElementType.ROAD]
    for lane_id, lane in lanes.items():
        src_road = lane_to_road[lane_id]
        for next_lane_id in lane.next_lanes:
            if next_lane_id not in lane_to_road:
                continue
            roads[src_road].next_roads.add(lane_to_road[next_lane_id])

    for road in roads.values():
        road.prev_roads.clear()
    for road in roads.values():
        for next_road_id in road.next_roads:
            roads[next_road_id].prev_roads.add(road.id)


def translate_lane(
    map_feature: waymo_map_pb2.MapFeature,
    boundaries: Dict[int, Polyline],
) -> RoadLane:
    lane: waymo_map_pb2.LaneCenter = map_feature.lane
    center = _centerline_polyline(lane)
    left_edge = _boundary_polyline(lane.left_boundaries, boundaries)
    right_edge = _boundary_polyline(lane.right_boundaries, boundaries)

    if left_edge is None:
        left_edge = _edge_from_center(center, 2.0)
    if right_edge is None:
        right_edge = _edge_from_center(center, -2.0)

    road_lane = RoadLane(
        id=str(map_feature.id),
        center=center,
        left_edge=left_edge,
        right_edge=right_edge,
    )
    road_lane.prev_lanes.update([str(eid) for eid in lane.entry_lanes])
    road_lane.next_lanes.update([str(eid) for eid in lane.exit_lanes])

    for neighbor in lane.left_neighbors:
        road_lane.adj_lanes_left.add(str(neighbor.feature_id))

    for neighbor in lane.right_neighbors:
        road_lane.adj_lanes_right.add(str(neighbor.feature_id))

    return road_lane


def extract_traffic_lights(
    dynamic_states: List[scenario_pb2.DynamicMapState],
) -> Dict[Tuple[str, int], TrafficLightStatus]:
    ret: Dict[Tuple[str, int], TrafficLightStatus] = {}
    for i, dynamic_state in enumerate(dynamic_states):
        for lane_state in dynamic_state.lane_states:
            ret[(str(lane_state.lane), i)] = translate_traffic_state(lane_state.state)

    return ret


def translate_traffic_state(
    state: waymo_map_pb2.TrafficSignalLaneState.State,
) -> TrafficLightStatus:
    # TODO(bivanovic): The traffic light type doesn't align between waymo and trajdata,
    # since trajdata's TrafficLightStatus does not include a yellow light yet.
    # For now, we set caution = red.
    if state in GREEN:
        return TrafficLightStatus.GREEN
    if state in RED:
        return TrafficLightStatus.RED
    return TrafficLightStatus.UNKNOWN


def interpolate_array(data: List) -> np.array:
    return pd.DataFrame(data).interpolate(limit_area="inside").to_numpy()
