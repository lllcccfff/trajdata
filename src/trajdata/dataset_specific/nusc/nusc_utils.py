from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from trajdata.data_structures import Agent, AgentMetadata, AgentType, FixedExtent, Scene
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    Road,
    RoadArea,
    RoadAreaPart,
    RoadLane,
)
from trajdata.utils import arr_utils, map_utils

NUSC_DT: Final[float] = 0.5


def frame_iterator(nusc_obj: NuScenes, scene: Scene) -> Dict[str, Union[str, int]]:
    """Loops through all frames in a scene and yields them for the caller to deal with the information."""
    curr_scene_token: str = scene.data_access_info["first_sample_token"]
    while curr_scene_token:
        frame = nusc_obj.get("sample", curr_scene_token)

        yield frame

        curr_scene_token = frame["next"]


def agent_iterator(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    """Loops through all annotations (agents) in a frame and yields them for the caller to deal with the information."""
    ann_token: str
    for ann_token in frame_info["anns"]:
        ann_record = nusc_obj.get("sample_annotation", ann_token)

        agent_category: str = ann_record["category_name"]
        if agent_category.startswith("vehicle") or agent_category.startswith("human"):
            yield ann_record


def get_ego_pose(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    cam_front_data = nusc_obj.get("sample_data", frame_info["data"]["CAM_FRONT"])
    ego_pose = nusc_obj.get("ego_pose", cam_front_data["ego_pose_token"])
    return ego_pose


def agg_agent_data(
    nusc_obj: NuScenes,
    agent_data: Dict[str, Any],
    curr_scene_index: int,
    frame_idx_dict: Dict[str, int],
) -> Agent:
    """Loops through all annotations of a specific agent in a scene and aggregates their data into an Agent object."""
    if agent_data["prev"]:
        print("WARN: This is not the first frame of this agent!")

    translation_list = [np.array(agent_data["translation"][:3])[np.newaxis]]
    agent_size = agent_data["size"]
    yaw_list = [Quaternion(agent_data["rotation"]).yaw_pitch_roll[0]]

    prev_idx: int = curr_scene_index
    curr_sample_ann_token: str = agent_data["next"]
    while curr_sample_ann_token:
        agent_data = nusc_obj.get("sample_annotation", curr_sample_ann_token)

        translation = np.array(agent_data["translation"][:3])
        heading = Quaternion(agent_data["rotation"]).yaw_pitch_roll[0]
        curr_idx: int = frame_idx_dict[agent_data["sample_token"]]
        if curr_idx > prev_idx + 1:
            fill_time = np.arange(prev_idx + 1, curr_idx)
            xs = np.interp(
                x=fill_time,
                xp=[prev_idx, curr_idx],
                fp=[translation_list[-1][0, 0], translation[0]],
            )
            ys = np.interp(
                x=fill_time,
                xp=[prev_idx, curr_idx],
                fp=[translation_list[-1][0, 1], translation[1]],
            )
            zs = np.interp(
                x=fill_time,
                xp=[prev_idx, curr_idx],
                fp=[translation_list[-1][0, 2], translation[2]],
            )
            headings: np.ndarray = arr_utils.angle_wrap(
                np.interp(
                    x=fill_time,
                    xp=[prev_idx, curr_idx],
                    fp=np.unwrap([yaw_list[-1], heading]),
                )
            )
            translation_list.append(np.stack([xs, ys, zs], axis=1))
            yaw_list.extend(headings.tolist())

        translation_list.append(translation[np.newaxis])
        # size_list.append(agent_data['size'])
        yaw_list.append(heading)

        prev_idx = curr_idx
        curr_sample_ann_token = agent_data["next"]

    translations_np = np.concatenate(translation_list, axis=0)

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos = translations_np[0, :2] - (
        translations_np[1, :2] - translations_np[0, :2]
    )
    velocities_np = (
        np.diff(
            translations_np[:, :2], axis=0, prepend=np.expand_dims(prepend_pos, axis=0)
        )
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    anno_yaws_np = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)
    # yaws_np = np.expand_dims(
    #     np.arctan2(velocities_np[:, 1], velocities_np[:, 0]), axis=1
    # )
    # sizes_np = np.stack(size_list, axis=0)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.plot(translations_np[:, 0], translations_np[:, 1], color="blue")
    # ax.quiver(
    #     translations_np[:, 0],
    #     translations_np[:, 1],
    #     np.cos(anno_yaws_np),
    #     np.sin(anno_yaws_np),
    #     color="green",
    #     label="annotated heading"
    # )
    # ax.quiver(
    #     translations_np[:, 0],
    #     translations_np[:, 1],
    #     np.cos(yaws_np),
    #     np.sin(yaws_np),
    #     color="orange",
    #     label="velocity heading"
    # )
    # ax.scatter([translations_np[0, 0]], [translations_np[0, 1]], color="red", label="Start", zorder=20)
    # ax.legend(loc='best')
    # plt.show()

    agent_data_np = np.concatenate(
        [translations_np, velocities_np, accelerations_np, anno_yaws_np], axis=1
    )
    last_timestep = curr_scene_index + agent_data_np.shape[0] - 1
    agent_data_df = pd.DataFrame(
        agent_data_np,
        columns=["x", "y", "z", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [
                (agent_data["instance_token"], idx)
                for idx in range(curr_scene_index, last_timestep + 1)
            ],
            names=["agent_id", "scene_ts"],
        ),
    )

    agent_type = nusc_type_to_unified_type(agent_data["category_name"])
    agent_metadata = AgentMetadata(
        name=agent_data["instance_token"],
        agent_type=agent_type,
        first_timestep=curr_scene_index,
        last_timestep=last_timestep,
        extent=FixedExtent(
            length=agent_size[1], width=agent_size[0], height=agent_size[2]
        ),
    )
    return Agent(
        metadata=agent_metadata,
        data=agent_data_df,
    )


def nusc_type_to_unified_type(nusc_type: str) -> AgentType:
    if nusc_type.startswith("human"):
        return AgentType.PEDESTRIAN
    elif nusc_type == "vehicle.bicycle":
        return AgentType.BICYCLE
    elif nusc_type == "vehicle.motorcycle":
        return AgentType.MOTORCYCLE
    elif nusc_type.startswith("vehicle"):
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


def agg_ego_data(nusc_obj: NuScenes, scene: Scene) -> Agent:
    translation_list: List[np.ndarray] = list()
    yaw_list: List[float] = list()
    for frame_info in frame_iterator(nusc_obj, scene):
        ego_pose = get_ego_pose(nusc_obj, frame_info)
        yaw_list.append(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
        translation_list.append(ego_pose["translation"])

    translations_np: np.ndarray = np.stack(translation_list, axis=0)

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos: np.ndarray = translations_np[0, :2] - (
        translations_np[1, :2] - translations_np[0, :2]
    )
    velocities_np: np.ndarray = (
        np.diff(
            translations_np[:, :2], axis=0, prepend=np.expand_dims(prepend_pos, axis=0)
        )
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel: np.ndarray = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np: np.ndarray = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    yaws_np: np.ndarray = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)
    # yaws_np = np.expand_dims(np.arctan2(velocities_np[:, 1], velocities_np[:, 0]), axis=1)

    ego_data_np: np.ndarray = np.concatenate(
        [translations_np, velocities_np, accelerations_np, yaws_np], axis=1
    )
    ego_data_df = pd.DataFrame(
        ego_data_np,
        columns=["x", "y", "z", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [("ego", idx) for idx in range(ego_data_np.shape[0])],
            names=["agent_id", "scene_ts"],
        ),
    )

    ego_metadata = AgentMetadata(
        name="ego",
        agent_type=AgentType.VEHICLE,
        first_timestep=0,
        last_timestep=ego_data_np.shape[0] - 1,
        extent=FixedExtent(length=4.084, width=1.730, height=1.562),
    )
    return Agent(
        metadata=ego_metadata,
        data=ego_data_df,
    )


def extract_lane_center(nusc_map: NuScenesMap, lane_record) -> np.ndarray:
    # Getting the lane center's points.
    curr_lane = nusc_map.arcline_path_3.get(lane_record["token"], [])
    lane_midline: np.ndarray = np.array(
        arcline_path_utils.discretize_lane(curr_lane, resolution_meters=0.5)
    )[:, :2]

    # For some reason, nuScenes duplicates a few entries
    # (likely how they're building their arcline representation).
    # We delete those duplicate entries here.
    duplicate_check: np.ndarray = np.where(
        np.linalg.norm(np.diff(lane_midline, axis=0, prepend=0), axis=1) < 1e-10
    )[0]
    if duplicate_check.size > 0:
        lane_midline = np.delete(lane_midline, duplicate_check, axis=0)

    return lane_midline


def extract_lane_and_edges(
    nusc_map: NuScenesMap, lane_record
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Getting the bounding polygon vertices.
    lane_polygon_obj = nusc_map.get("polygon", lane_record["polygon_token"])
    polygon_nodes = [
        nusc_map.get("node", node_token)
        for node_token in lane_polygon_obj["exterior_node_tokens"]
    ]
    polygon_pts: np.ndarray = np.array(
        [(node["x"], node["y"]) for node in polygon_nodes]
    )

    # Getting the lane center's points.
    lane_midline: np.ndarray = extract_lane_center(nusc_map, lane_record)

    # Computing the closest lane center point to each bounding polygon vertex.
    closest_midlane_pt: np.ndarray = np.argmin(cdist(polygon_pts, lane_midline), axis=1)
    # Computing the local direction of the lane at each lane center point.
    direction_vectors: np.ndarray = np.diff(
        lane_midline,
        axis=0,
        prepend=lane_midline[[0]] - (lane_midline[[1]] - lane_midline[[0]]),
    )

    # Selecting the direction vectors at the closest lane center point per polygon vertex.
    local_dir_vecs: np.ndarray = direction_vectors[closest_midlane_pt]
    # Calculating the vectors from the the closest lane center point per polygon vertex to the polygon vertex.
    origin_to_polygon_vecs: np.ndarray = polygon_pts - lane_midline[closest_midlane_pt]

    # Computing the perpendicular dot product.
    # See https://www.xarg.org/book/linear-algebra/2d-perp-product/
    # If perp_dot_product < 0, then the associated polygon vertex is
    # on the right edge of the lane.
    perp_dot_product: np.ndarray = (
        local_dir_vecs[:, 0] * origin_to_polygon_vecs[:, 1]
        - local_dir_vecs[:, 1] * origin_to_polygon_vecs[:, 0]
    )

    # Determining which indices are on the right of the lane center.
    on_right: np.ndarray = perp_dot_product < 0
    # Determining the boundary between the left/right polygon vertices
    # (they will be together in blocks due to the ordering of the polygon vertices).
    idx_changes: int = np.where(np.roll(on_right, 1) < on_right)[0].item()

    if idx_changes > 0:
        # If the block of left/right points spreads across the bounds of the array,
        # roll it until the boundary between left/right points is at index 0.
        # This is important so that the following index selection orders points
        # without jumps.
        polygon_pts = np.roll(polygon_pts, shift=-idx_changes, axis=0)
        on_right = np.roll(on_right, shift=-idx_changes)

    left_pts: np.ndarray = polygon_pts[~on_right]
    right_pts: np.ndarray = polygon_pts[on_right]

    # Final ordering check, ensuring that left_pts and right_pts can be combined
    # into a polygon without the endpoints intersecting.
    # Reversing the one lane edge that does not match the ordering of the midline.
    if map_utils.endpoints_intersect(left_pts, right_pts):
        if not map_utils.order_matches(left_pts, lane_midline):
            left_pts = left_pts[::-1]
        else:
            right_pts = right_pts[::-1]

    # Ensuring that left and right have the same number of points.
    # This is necessary, not for data storage but for later rasterization.
    if left_pts.shape[0] < right_pts.shape[0]:
        left_pts = map_utils.interpolate(left_pts, num_pts=right_pts.shape[0])
    elif right_pts.shape[0] < left_pts.shape[0]:
        right_pts = map_utils.interpolate(right_pts, num_pts=left_pts.shape[0])

    return (
        lane_midline,
        left_pts,
        right_pts,
    )


def extract_area(nusc_map: NuScenesMap, area_record) -> np.ndarray:
    token_key: str
    if "exterior_node_tokens" in area_record:
        token_key = "exterior_node_tokens"
    elif "node_tokens" in area_record:
        token_key = "node_tokens"

    polygon_nodes = [
        nusc_map.get("node", node_token) for node_token in area_record[token_key]
    ]

    return np.array([(node["x"], node["y"]) for node in polygon_nodes])


def _assign_lanes_to_roads(nusc_map: NuScenesMap, lane_ids: Set[str]) -> Dict[str, str]:
    road_block_ids = [road_record["token"] for road_record in nusc_map.road_block]
    road_block_polygons = [
        nusc_map.extract_polygon(road_record["polygon_token"])
        for road_record in nusc_map.road_block
    ]
    road_block_tree = STRtree(road_block_polygons)

    lane_to_road = {}
    for lane_id in lane_ids:
        lane_record = nusc_map.get("lane", lane_id)
        centerline = extract_lane_center(nusc_map, lane_record)
        road_block_counts: Dict[str, int] = {}
        for center_point in centerline:
            point = Point(float(center_point[0]), float(center_point[1]))
            for road_block_idx in sorted(road_block_tree.query(point)):
                if point.within(road_block_polygons[road_block_idx]):
                    road_id = road_block_ids[road_block_idx]
                    road_block_counts[road_id] = road_block_counts.get(road_id, 0) + 1
                    break

        if not road_block_counts:
            continue

        road_id, count = max(road_block_counts.items(), key=lambda item: item[1])
        if count / len(centerline) >= 0.4:
            lane_to_road[lane_id] = road_id
    return lane_to_road


def _connector_road_id(prev_road_id: str, next_road_id: str) -> str:
    return f"lane_connector:{prev_road_id}:{next_road_id}"


def _boundary_connector_road_id(lane_id: str) -> str:
    return f"lane_connector:boundary:{lane_id}"


def _assign_lane_connectors_to_connector_roads(
    vector_map: VectorMap,
    lane_connector_ids: Set[str],
    lane_to_road: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, Tuple[Optional[str], Optional[str]]]]:
    connector_lane_to_road: Dict[str, str] = {}
    connector_road_pairs: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for lane_id in lane_connector_ids:
        lane = vector_map.get_road_lane(lane_id)
        prev_road_ids = {
            lane_to_road[prev_lane_id]
            for prev_lane_id in lane.prev_lanes
            if prev_lane_id in lane_to_road
        }
        next_road_ids = {
            lane_to_road[next_lane_id]
            for next_lane_id in lane.next_lanes
            if next_lane_id in lane_to_road
        }
        if len(prev_road_ids) > 1 or len(next_road_ids) > 1:
            raise RuntimeError(
                f"NuScenes lane connector {lane_id} must map to at most one previous "
                f"road and one next road, got {prev_road_ids} -> {next_road_ids}."
            )

        prev_road_id = next(iter(prev_road_ids), None)
        next_road_id = next(iter(next_road_ids), None)
        connector_road_id = (
            _connector_road_id(prev_road_id, next_road_id)
            if prev_road_id is not None and next_road_id is not None
            else _boundary_connector_road_id(lane_id)
        )
        connector_lane_to_road[lane_id] = connector_road_id
        connector_road_pairs[connector_road_id] = (prev_road_id, next_road_id)

    return connector_lane_to_road, connector_road_pairs


def _road_successors_from_lanes(
    vector_map: VectorMap, lane_to_road: Dict[str, str]
) -> Dict[str, Set[str]]:
    road_successors: Dict[str, Set[str]] = {}
    vector_lane_ids = set(vector_map.elements[MapElementType.ROAD_LANE].keys())
    for lane_id, road_id in lane_to_road.items():
        lane = vector_map.get_road_lane(lane_id)
        for next_lane_id in lane.next_lanes:
            if next_lane_id not in vector_lane_ids:
                continue
            next_road_id = lane_to_road[next_lane_id]
            if next_road_id != road_id:
                road_successors.setdefault(road_id, set()).add(next_road_id)
    return road_successors


def _lane_divider_node_key(lane_record: dict, side: str) -> Tuple[str, ...]:
    return tuple(
        segment["node_token"]
        for segment in lane_record[f"{side}_lane_divider_segments"]
    )


def _add_lane_adjacency_within_roads(
    vector_map: VectorMap, nusc_map: NuScenesMap
) -> None:
    lane_records = {lane_record["token"]: lane_record for lane_record in nusc_map.lane}
    left_divider_to_lanes: Dict[Tuple[str, ...], List[str]] = {}
    right_divider_to_lanes: Dict[Tuple[str, ...], List[str]] = {}
    for lane_record in lane_records.values():
        lane_id = lane_record["token"]
        left_key = _lane_divider_node_key(lane_record, "left")
        right_key = _lane_divider_node_key(lane_record, "right")
        if left_key:
            left_divider_to_lanes.setdefault(left_key, []).append(lane_id)
        if right_key:
            right_divider_to_lanes.setdefault(right_key, []).append(lane_id)

    for road in vector_map.elements[MapElementType.ROAD].values():
        road_lane_ids = set(road.lane_ids)
        for lane_id in road_lane_ids.intersection(lane_records):
            lane_record = lane_records[lane_id]
            lane = vector_map.get_road_lane(lane_id)

            left_key = _lane_divider_node_key(lane_record, "left")
            if left_key:
                lane.adj_lanes_left.update(
                    candidate_id
                    for candidate_id in right_divider_to_lanes.get(left_key, [])
                    if candidate_id in road_lane_ids and candidate_id != lane_id
                )

            right_key = _lane_divider_node_key(lane_record, "right")
            if right_key:
                lane.adj_lanes_right.update(
                    candidate_id
                    for candidate_id in left_divider_to_lanes.get(right_key, [])
                    if candidate_id in road_lane_ids and candidate_id != lane_id
                )


def _road_lane_area_polygon(lane: RoadLane) -> Polygon:
    if lane.left_edge is None or lane.right_edge is None:
        raise RuntimeError(f"Road lane {lane.id} is missing lane boundaries.")

    shell = np.concatenate([lane.left_edge.xy, lane.right_edge.xy[::-1]], axis=0)
    lane_area = Polygon(shell)
    if lane_area.is_empty or not lane_area.is_valid:
        raise RuntimeError(f"Road lane {lane.id} has an invalid lane area polygon.")
    return lane_area


def _polygon_to_road_area(
    road_id: str, geometry: Union[Polygon, MultiPolygon]
) -> RoadArea:
    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = sorted(geometry.geoms, key=lambda polygon: polygon.wkb)
    else:
        raise TypeError(
            f"Road {road_id} has unsupported geometry {geometry.geom_type}."
        )

    areas: List[RoadAreaPart] = list()
    for polygon in polygons:
        exterior = np.asarray(polygon.exterior.coords[:-1], dtype=float)
        if exterior.shape[0] < 3:
            raise RuntimeError(f"Road area {road_id} has fewer than three points.")

        area = RoadAreaPart(exterior_polygon=Polyline(exterior))
        for interior in polygon.interiors:
            hole = np.asarray(interior.coords[:-1], dtype=float)
            if hole.shape[0] < 3:
                raise RuntimeError(f"Road area {road_id} has an invalid hole.")
            area.interior_holes.append(Polyline(hole))
        areas.append(area)

    return RoadArea(id=road_id, areas=areas)


def _add_connector_road_area(
    vector_map: VectorMap, connector_road_id: str, lane_ids: Set[str]
) -> None:
    if connector_road_id in vector_map.elements[MapElementType.ROAD_AREA]:
        raise RuntimeError(f"Road area {connector_road_id} already exists.")

    connector_area = unary_union(
        [
            _road_lane_area_polygon(vector_map.get_road_lane(lane_id))
            for lane_id in sorted(lane_ids)
        ]
    )
    vector_map.add_map_element(
        _polygon_to_road_area(connector_road_id, connector_area)
    )


def add_nusc_road_topology(
    vector_map: VectorMap, nusc_map: NuScenesMap
) -> None:
    """Adds NuScenes road topology to an already lane-populated VectorMap."""
    lane_ids = set(vector_map.elements[MapElementType.ROAD_LANE].keys())
    lane_layer_ids = {
        lane_record["token"]
        for lane_record in nusc_map.lane
        if lane_record["token"] in lane_ids
    }
    lane_connector_layer_ids = {
        lane_record["token"]
        for lane_record in nusc_map.lane_connector
        if lane_record["token"] in lane_ids
    }
    lane_to_road = _assign_lanes_to_roads(nusc_map, lane_layer_ids)
    discarded_lane_ids = lane_layer_ids.difference(lane_to_road)
    for lane_id in discarded_lane_ids:
        del vector_map.elements[MapElementType.ROAD_LANE][lane_id]
    lane_ids.difference_update(discarded_lane_ids)

    connector_lane_to_road, connector_road_pairs = (
        _assign_lane_connectors_to_connector_roads(
            vector_map, lane_connector_layer_ids, lane_to_road
        )
    )
    lane_to_road.update(connector_lane_to_road)
    road_ids = set(lane_to_road.values())
    missing_lane_ids = lane_ids.difference(lane_to_road)
    if missing_lane_ids:
        raise RuntimeError(
            f"NuScenes road topology did not cover {len(missing_lane_ids)} lanes "
            f"in map {vector_map.map_id}."
        )

    road_to_lanes: Dict[str, Set[str]] = {}
    for lane_id, road_id in lane_to_road.items():
        road_to_lanes.setdefault(road_id, set()).add(lane_id)

    road_successors: Dict[str, Set[str]] = {road_id: set() for road_id in road_ids}
    for road_id, next_road_ids in _road_successors_from_lanes(
        vector_map, lane_to_road
    ).items():
        road_successors[road_id].update(next_road_ids)

    vector_map.elements[MapElementType.ROAD].clear()
    vector_map.lane_to_road.clear()
    vector_map.roads = None

    road_records = {
        road_record["token"]: road_record for road_record in nusc_map.road_block
    }
    connector_road_ids = set(connector_road_pairs)
    for road_id in sorted(road_ids):
        if road_id in road_records:
            _add_nusc_road_area(vector_map, nusc_map, road_records[road_id])
        elif road_id in connector_road_ids:
            _add_connector_road_area(vector_map, road_id, road_to_lanes[road_id])
        else:
            raise RuntimeError(
                f"NuScenes road {road_id} has no road block or lane connector."
            )

        vector_map.add_map_element(
            Road(
                id=road_id,
                road_area_id=road_id,
                lane_ids=road_to_lanes[road_id],
                next_roads=road_successors[road_id],
            )
        )

    for road in vector_map.elements[MapElementType.ROAD].values():
        road.prev_roads.clear()
    for road in vector_map.elements[MapElementType.ROAD].values():
        for next_road_id in road.next_roads:
            next_road = vector_map.elements[MapElementType.ROAD].get(next_road_id)
            if next_road is not None:
                next_road.prev_roads.add(road.id)

    _add_lane_adjacency_within_roads(vector_map, nusc_map)


def _connector_edges_from_center(center_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    headings = map_utils.get_polyline_headings(center_pts)[:, 0]
    normal = np.stack([-np.sin(headings), np.cos(headings)], axis=1)
    left_pts = center_pts + normal * 2.0
    right_pts = center_pts - normal * 2.0
    return left_pts, right_pts


def _add_nusc_lane_connector(
    vector_map: VectorMap, nusc_map: NuScenesMap, lane_record: dict
) -> None:
    lane_record_token: str = lane_record["token"]
    center_pts = extract_lane_center(nusc_map, lane_record)
    left_pts, right_pts = _connector_edges_from_center(center_pts)
    new_lane = RoadLane(
        id=lane_record_token,
        center=Polyline(center_pts),
        left_edge=Polyline(left_pts),
        right_edge=Polyline(right_pts),
    )

    new_lane.prev_lanes.update(nusc_map.get_incoming_lane_ids(lane_record_token))
    new_lane.next_lanes.update(nusc_map.get_outgoing_lane_ids(lane_record_token))
    vector_map.add_map_element(new_lane)


def _add_nusc_road_area(
    vector_map: VectorMap, nusc_map: NuScenesMap, road_record: dict
) -> None:
    road_id = road_record["token"]
    if road_id in vector_map.elements[MapElementType.ROAD_AREA]:
        return

    polygon_pts = extract_area(nusc_map, road_record)
    road_area_part = RoadAreaPart(exterior_polygon=Polyline(polygon_pts))
    for hole in road_record["holes"]:
        hole_pts = extract_area(nusc_map, hole)
        road_area_part.interior_holes.append(Polyline(hole_pts))
    vector_map.add_map_element(RoadArea(id=road_id, areas=[road_area_part]))


def populate_vector_map(vector_map: VectorMap, nusc_map: NuScenesMap) -> None:
    # Setting the map bounds.
    vector_map.extent = np.array(
        [
            nusc_map.explorer.canvas_min_x,
            nusc_map.explorer.canvas_min_y,
            0.0,
            nusc_map.explorer.canvas_max_x,
            nusc_map.explorer.canvas_max_y,
            0.0,
        ]
    )

    overall_pbar = tqdm(
        total=len(nusc_map.lane)
        + len(nusc_map.lane_connector)
        + len(nusc_map.road_block)
        + len(nusc_map.road_segment)
        + len(nusc_map.ped_crossing)
        + len(nusc_map.walkway),
        desc=f"Getting {nusc_map.map_name} Elements",
        position=1,
        leave=False,
    )

    for lane_record in nusc_map.lane:
        center_pts, left_pts, right_pts = extract_lane_and_edges(nusc_map, lane_record)

        lane_record_token: str = lane_record["token"]

        new_lane = RoadLane(
            id=lane_record_token,
            center=Polyline(center_pts),
            left_edge=Polyline(left_pts),
            right_edge=Polyline(right_pts),
        )

        for lane_id in nusc_map.get_incoming_lane_ids(lane_record_token):
            # Need to do this because some incoming/outgoing lane_connector IDs
            # do not exist as lane_connectors...
            if lane_id in nusc_map._token2ind["lane_connector"]:
                new_lane.prev_lanes.add(lane_id)

        for lane_id in nusc_map.get_outgoing_lane_ids(lane_record_token):
            # Need to do this because some incoming/outgoing lane_connector IDs
            # do not exist as lane_connectors...
            if lane_id in nusc_map._token2ind["lane_connector"]:
                new_lane.next_lanes.add(lane_id)

        # new_lane.adjacent_lanes_left.append(
        #     l5_lane.adjacent_lane_change_left.id
        # )
        # new_lane.adjacent_lanes_right.append(
        #     l5_lane.adjacent_lane_change_right.id
        # )

        # Adding the element to the map.
        vector_map.add_map_element(new_lane)
        overall_pbar.update()

    for lane_record in nusc_map.lane_connector:
        _add_nusc_lane_connector(vector_map, nusc_map, lane_record)
        overall_pbar.update()

    add_nusc_road_topology(vector_map, nusc_map)

    for road_record in nusc_map.road_block:
        overall_pbar.update()

    for road_record in nusc_map.road_segment:
        if not road_record["is_intersection"]:
            overall_pbar.update()
            continue
        overall_pbar.update()

    for ped_area_record in nusc_map.ped_crossing:
        polygon_pts = extract_area(nusc_map, ped_area_record)

        # Adding the element to the map.
        vector_map.add_map_element(
            PedCrosswalk(id=ped_area_record["token"], polygon=Polyline(polygon_pts))
        )
        overall_pbar.update()

    for ped_area_record in nusc_map.walkway:
        polygon_pts = extract_area(nusc_map, ped_area_record)

        # Adding the element to the map.
        vector_map.add_map_element(
            PedWalkway(id=ped_area_record["token"], polygon=Polyline(polygon_pts))
        )
        overall_pbar.update()

    overall_pbar.close()
