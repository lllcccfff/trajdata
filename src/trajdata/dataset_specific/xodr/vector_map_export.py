"""VectorMap export functionality for XODR parsed data.

This module converts parsed OpenDRIVE data into trajdata VectorMap format.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import numpy as np
from trajdata.maps import vec_map
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

from .datatypes import LaneGeom
from .geo_transform import apply_transform
from .parser import parse_xodr

# Constants
# Maximum distance (meters) between interpolated polyline points
MAX_POLYLINE_POINT_DIST = 2.0


def populate_vector_map_from_xodr(
    vmap: vec_map.VectorMap,
    xodr_str: str,
    *,
    resolution: float = 0.5,
    t_xodr_enu_to_sim: Optional[np.ndarray] = None,
) -> None:
    """Populate a VectorMap with data parsed from an OpenDRIVE XML string.

    This function parses the XODR data and converts it into trajdata's
    VectorMap format, optionally applying a coordinate transformation.

    Args:
        vmap: Empty VectorMap instance to populate. map_id should be
              set by the caller.
        xodr_str: OpenDRIVE XML content as string
        resolution: Optional sampling resolution along road centerlines in meters.
                   Defaults to 0.5 meters.
        t_xodr_enu_to_sim: Optional 4x4 SE(3) matrix mapping XODR ENU coords
                            to sim space. If provided, every xyz point is
                            transformed before insertion.
    """
    # Parse the XODR data
    parsed = parse_xodr(xodr_str, resolution=resolution)

    # Create partial function for coordinate transformation
    transform_points = partial(apply_transform, transform_mat=t_xodr_enu_to_sim)

    # Convert lanes to VectorMap format
    lane_id_map = _add_lanes_to_vmap(
        vmap, parsed, transform_points, MAX_POLYLINE_POINT_DIST
    )
    _add_directed_roads_to_vmap(vmap, parsed, lane_id_map, transform_points)

    # Convert sidewalks to VectorMap format
    _add_sidewalks_to_vmap(vmap, parsed, transform_points, MAX_POLYLINE_POINT_DIST)

    # Convert crosswalks to VectorMap format
    _add_crosswalks_to_vmap(vmap, parsed, transform_points, MAX_POLYLINE_POINT_DIST)

    # Set the map extent, transforming if necessary
    vmap.extent = _transform_extent(
        parsed.extent, transform_points, t_xodr_enu_to_sim
    )


def _transform_extent(extent, transform_fn, t_xodr_enu_to_sim):
    """Transform a 3D bounding box extent if a transformation is provided.

    When coordinates are transformed (e.g., rotated), the axis-aligned bounding box
    changes. This function transforms all 8 corners of the original box and
    computes the new axis-aligned bounds.

    Args:
        extent: Original extent as [min_x, min_y, min_z, max_x, max_y, max_z]
        transform_fn: Function to transform coordinates
        t_xodr_enu_to_sim: Transformation matrix from XODR ENU to sim space
                            (if None, returns original extent)

    Returns:
        Transformed extent in same format as input
    """
    if t_xodr_enu_to_sim is None:
        return extent

    # Extract min and max corners
    min_corner = extent[:3]
    max_corner = extent[3:]

    # Create all 8 corners of the bounding box by taking all combinations
    # of min/max values for each dimension
    corners = np.array(
        [
            [x, y, z]
            for x in [min_corner[0], max_corner[0]]
            for y in [min_corner[1], max_corner[1]]
            for z in [min_corner[2], max_corner[2]]
        ]
    )

    # Transform all corners
    transformed_corners = transform_fn(corners)

    # Find new axis-aligned bounding box
    new_min = np.min(transformed_corners, axis=0)
    new_max = np.max(transformed_corners, axis=0)

    return np.concatenate([new_min, new_max])


def _add_lanes_to_vmap(vmap, parsed, transform_fn, max_dist) -> Dict[str, str]:
    """Add lane geometries to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
        max_dist: Maximum distance between interpolated points
    """
    # Build a stable numeric ID mapping for compatibility with trajdata
    # We assign incremental integers starting from 0 in deterministic order
    # The IDs are stored as strings so that downstream code that casts to
    # int(float(id)) still succeeds, replicating the clipgt lane-id convention
    driving_lanes = [lane for lane in parsed.lanes.values() if lane.is_driving]
    lane_id_map = {lane.unique_id: str(i) for i, lane in enumerate(driving_lanes)}

    for lg in driving_lanes:
        if lg.left_edge is None or lg.right_edge is None:
            raise ValueError(f"XODR RoadLane {lg.unique_id} is missing lane edges")
        center_pl = Polyline(transform_fn(lg.center)).interpolate(max_dist=max_dist)
        left_pl = Polyline(transform_fn(lg.left_edge)).interpolate(max_dist=max_dist)
        right_pl = Polyline(transform_fn(lg.right_edge)).interpolate(max_dist=max_dist)

        # Positive standard-direction lanes run opposite the reference line, so
        # geometric left/right must be swapped into the lane's travel frame.
        source_left = lg.adj_left if lg.lane_id_xml < 0 else lg.adj_right
        source_right = lg.adj_right if lg.lane_id_xml < 0 else lg.adj_left
        adj_left = {lane_id_map[x] for x in source_left if x in lane_id_map}
        adj_right = {lane_id_map[x] for x in source_right if x in lane_id_map}
        next_lanes = {lane_id_map[x] for x in lg.next_lanes if x in lane_id_map}
        prev_lanes = {lane_id_map[x] for x in lg.prev_lanes if x in lane_id_map}
        lane_elem = RoadLane(
            id=lane_id_map[lg.unique_id],
            center=center_pl,
            left_edge=left_pl,
            right_edge=right_pl,
            adj_lanes_left=adj_left,
            adj_lanes_right=adj_right,
            next_lanes=next_lanes,
            prev_lanes=prev_lanes,
        )
        vmap.add_map_element(lane_elem)
    return lane_id_map


def _directed_road_id(lane: LaneGeom) -> str:
    direction = "forward" if lane.lane_id_xml < 0 else "reverse"
    return f"{lane.road_id}:{direction}"


def _closed_road_area(inner_edge: np.ndarray, road_edge: np.ndarray) -> Polyline:
    if inner_edge.shape[0] < 2 or road_edge.shape[0] < 2:
        raise ValueError("XODR RoadArea boundaries must each contain at least two points")
    if not np.isfinite(inner_edge).all() or not np.isfinite(road_edge).all():
        raise ValueError("XODR RoadArea boundaries contain non-finite coordinates")

    aligned_endpoint_distance = np.linalg.norm(
        inner_edge[0] - road_edge[0]
    ) + np.linalg.norm(inner_edge[-1] - road_edge[-1])
    reversed_endpoint_distance = np.linalg.norm(
        inner_edge[0] - road_edge[-1]
    ) + np.linalg.norm(inner_edge[-1] - road_edge[0])
    if reversed_endpoint_distance < aligned_endpoint_distance:
        road_edge = road_edge[::-1]
    points = np.concatenate((inner_edge, road_edge[::-1]), axis=0)
    if not np.allclose(points[0], points[-1]):
        points = np.concatenate((points, points[[0]]), axis=0)
    return Polyline(points)


def _add_directed_roads_to_vmap(
    vmap,
    parsed,
    lane_id_map: Dict[str, str],
    transform_fn,
) -> None:
    road_lanes: Dict[str, List[LaneGeom]] = defaultdict(list)
    lane_to_road: Dict[str, str] = {}
    for lane in parsed.lanes.values():
        if lane.unique_id not in lane_id_map:
            continue
        road_id = _directed_road_id(lane)
        road_lanes[road_id].append(lane)
        lane_to_road[lane.unique_id] = road_id

    for road_id, lanes in sorted(road_lanes.items()):
        lane_signs = {int(np.sign(lane.lane_id_xml)) for lane in lanes}
        if len(lane_signs) != 1:
            raise ValueError(f"Directed XODR Road {road_id} mixes lane directions")

        innermost_lane = min(lanes, key=lambda lane: abs(lane.lane_id_xml))
        if innermost_lane.left_edge is None:
            raise ValueError(f"Directed XODR Road {road_id} has no inner lane edge")
        side = "R" if innermost_lane.lane_id_xml < 0 else "L"
        edge_id = f"{innermost_lane.road_id}_{side}"
        if edge_id not in parsed.road_edges:
            raise KeyError(f"Directed XODR Road {road_id} has no RoadEdge {edge_id}")

        inner_edge = transform_fn(innermost_lane.left_edge)
        road_edge = transform_fn(parsed.road_edges[edge_id])
        vmap.add_map_element(
            RoadArea(
                id=road_id,
                areas=[
                    RoadAreaPart(
                        exterior_polygon=_closed_road_area(inner_edge, road_edge)
                    )
                ],
            )
        )
        vmap.add_map_element(
            Road(
                id=road_id,
                road_area_id=road_id,
                lane_ids={lane_id_map[lane.unique_id] for lane in lanes},
                prev_roads=set(),
                next_roads=set(),
            )
        )

    roads = vmap.elements[MapElementType.ROAD]
    for lane in parsed.lanes.values():
        source_road_id = lane_to_road.get(lane.unique_id)
        if source_road_id is None:
            continue
        for next_lane_id in lane.next_lanes:
            next_road_id = lane_to_road.get(next_lane_id)
            if next_road_id is not None and next_road_id != source_road_id:
                roads[source_road_id].next_roads.add(next_road_id)

    for road in roads.values():
        for next_road_id in road.next_roads:
            roads[next_road_id].prev_roads.add(road.id)

    if len(vmap.lane_to_road) != len(lane_id_map):
        raise RuntimeError(
            f"XODR Road topology covers {len(vmap.lane_to_road)} of "
            f"{len(lane_id_map)} RoadLanes"
        )


def _add_sidewalks_to_vmap(vmap, parsed, transform_fn, max_dist):
    """Add sidewalk (PedWalkway) geometries to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
        max_dist: Maximum distance between interpolated points
    """
    for sidewalk_id, polygon_points in parsed.sidewalks.items():
        sidewalk_elem = PedWalkway(
            id=sidewalk_id,
            polygon=Polyline(transform_fn(polygon_points)).interpolate(
                max_dist=max_dist
            ),
        )
        vmap.add_map_element(sidewalk_elem)


def _add_crosswalks_to_vmap(vmap, parsed, transform_fn, max_dist):
    """Add crosswalk (PedCrosswalk) geometries to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
        max_dist: Maximum distance between interpolated points
    """
    for crosswalk_id, polygon_points in parsed.crosswalks.items():
        crosswalk_elem = PedCrosswalk(
            id=crosswalk_id,
            polygon=Polyline(transform_fn(polygon_points)).interpolate(
                max_dist=max_dist
            ),
        )
        vmap.add_map_element(crosswalk_elem)
