from shapely.geometry import Polygon

from trajdata.dataset_specific.xodr.vector_map_export import (
    populate_vector_map_from_xodr,
)
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import MapElementType


def _lane_xml(lane_id: int, predecessor: bool, successor: bool) -> str:
    links = []
    if predecessor:
        links.append(f'<predecessor id="{lane_id}"/>')
    if successor:
        links.append(f'<successor id="{lane_id}"/>')
    return f"""
        <lane id="{lane_id}" type="driving">
          <link>{''.join(links)}</link>
          <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
        </lane>
    """


def _lane_section_xml(
    s: float, *, predecessor: bool, successor: bool
) -> str:
    left = "".join(
        _lane_xml(lane_id, predecessor, successor) for lane_id in (1, 2)
    )
    right = "".join(
        _lane_xml(lane_id, predecessor, successor) for lane_id in (-1, -2)
    )
    return f"""
      <laneSection s="{s}">
        <center><lane id="0" type="none"><link/></lane></center>
        <left>{left}</left>
        <right>{right}</right>
      </laneSection>
    """


def _road_xml(
    road_id: int,
    x: float,
    length: float,
    sections: str,
    road_link: str,
) -> str:
    return f"""
    <road name="" length="{length}" id="{road_id}" junction="-1">
      <link>{road_link}</link>
      <planView>
        <geometry s="0" x="{x}" y="0" hdg="0" length="{length}"><line/></geometry>
      </planView>
      <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
      <lanes>{sections}</lanes>
    </road>
    """


def _two_road_xodr() -> str:
    road_0_sections = "".join(
        [
            _lane_section_xml(0.0, predecessor=False, successor=True),
            _lane_section_xml(10.0, predecessor=True, successor=True),
        ]
    )
    road_1_sections = _lane_section_xml(
        0.0, predecessor=True, successor=False
    )
    road_0 = _road_xml(
        0,
        0.0,
        20.0,
        road_0_sections,
        '<successor elementType="road" elementId="1" contactPoint="start"/>',
    )
    road_1 = _road_xml(
        1,
        20.0,
        10.0,
        road_1_sections,
        '<predecessor elementType="road" elementId="0" contactPoint="end"/>',
    )
    return f'<OpenDRIVE>{road_0}{road_1}</OpenDRIVE>'


def test_xodr_exports_directed_lane_section_roads() -> None:
    vector_map = VectorMap("xodr:test")
    populate_vector_map_from_xodr(vector_map, _two_road_xodr(), resolution=1.0)

    lanes = vector_map.elements[MapElementType.ROAD_LANE]
    roads = vector_map.elements[MapElementType.ROAD]
    road_areas = vector_map.elements[MapElementType.ROAD_AREA]

    assert len(lanes) == 12
    assert set(roads) == {
        "0:0:forward",
        "0:0:reverse",
        "0:1:forward",
        "0:1:reverse",
        "1:0:forward",
        "1:0:reverse",
    }
    assert set(road_areas) == set(roads)
    assert len(vector_map.lane_to_road) == len(lanes)

    assert roads["0:0:forward"].next_roads == {"0:1:forward"}
    assert roads["0:1:forward"].next_roads == {"1:0:forward"}
    assert roads["1:0:reverse"].next_roads == {"0:1:reverse"}
    assert roads["0:1:reverse"].next_roads == {"0:0:reverse"}

    for road in roads.values():
        assert len(road.lane_ids) == 2
        assert road.road_area_id == road.id
        assert road.prev_roads <= set(roads)
        assert road.next_roads <= set(roads)
        area = road_areas[road.road_area_id]
        assert len(area.areas) == 1
        polygon = Polygon(area.areas[0].exterior_polygon.xy)
        assert polygon.is_valid
        assert polygon.area > 0.0

    for lane in lanes.values():
        assert lane.center is not None
        assert lane.left_edge is not None
        assert lane.right_edge is not None
        assert lane.adj_lanes_left <= set(lanes)
        assert lane.adj_lanes_right <= set(lanes)
        assert lane.prev_lanes <= set(lanes)
        assert lane.next_lanes <= set(lanes)
        assert vector_map.road_for_lane(lane.id) is not None
