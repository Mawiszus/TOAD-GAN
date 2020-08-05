from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.


FLOOR_TOKENS = OrderedDict(
    {
        "-": "grass",
     }
)

WALL_TOKENS = OrderedDict(
    {
        "W": "wall",
    }
)

ROAD_TOKENS = OrderedDict(
    {
        "R": "road",
     }
)

SPECIAL_TOKENS = OrderedDict(
    {
        "S": "starting line",
        "Q": "question block",
        "O": "oil",
        "C": "coin",
        "<": "boost",
     }
)

TOKEN_DOWNSAMPLING_HIERARCHY = [
    FLOOR_TOKENS,
    WALL_TOKENS,
    ROAD_TOKENS,
    SPECIAL_TOKENS,
]

TOKENS = OrderedDict(
    {**FLOOR_TOKENS, **WALL_TOKENS, **ROAD_TOKENS, **SPECIAL_TOKENS}
)

TOKEN_GROUPS = [FLOOR_TOKENS, WALL_TOKENS, ROAD_TOKENS, SPECIAL_TOKENS]

REPLACE_TOKENS = {}

