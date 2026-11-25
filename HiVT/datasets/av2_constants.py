# Copyright (c) 2022, Zikang Zhou. All rights reserved.
# Extended for Argoverse 2 class / map semantics.
from typing import Dict

from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.lane_segment import LaneMarkType, LaneType

# Dynamic actors used by the motion predictor (static classes are ignored).
DYNAMIC_OBJECT_TYPES = (
    ObjectType.VEHICLE,
    ObjectType.PEDESTRIAN,
    ObjectType.MOTORCYCLIST,
    ObjectType.CYCLIST,
    ObjectType.BUS,
    ObjectType.UNKNOWN,
)

ACTOR_TYPE_TO_ID: Dict[ObjectType, int] = {
    ObjectType.VEHICLE: 0,
    ObjectType.PEDESTRIAN: 1,
    ObjectType.MOTORCYCLIST: 2,
    ObjectType.CYCLIST: 3,
    ObjectType.BUS: 4,
    ObjectType.UNKNOWN: 5,
}
NUM_ACTOR_TYPES = len(ACTOR_TYPE_TO_ID)

# Map element semantics for ALEncoder (AV2).
# 0-2: lane types, 3: pedestrian crosswalk polyline.
LANE_TYPE_TO_ID: Dict[LaneType, int] = {
    LaneType.VEHICLE: 0,
    LaneType.BIKE: 1,
    LaneType.BUS: 2,
}
CROSSWALK_TYPE_ID = 3
NUM_LANE_TYPES = 4  # VEHICLE / BIKE / BUS / CROSSWALK

MARK_TYPE_TO_ID: Dict[LaneMarkType, int] = {
    mark_type: idx for idx, mark_type in enumerate(LaneMarkType)
}
NUM_MARK_TYPES = len(MARK_TYPE_TO_ID)

AV2_HISTORICAL_STEPS = 50
AV2_FUTURE_STEPS = 60
AV2_TOTAL_STEPS = AV2_HISTORICAL_STEPS + AV2_FUTURE_STEPS
AV_TRACK_ID = 'AV'
