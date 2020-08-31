from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.

VOID_TOKENS = OrderedDict(
    {
        "-": "Void",
     }
)

WALL_TOKENS = OrderedDict(
    {
        "D": "Door",
        "W": "Wall",
     }
)

FLOOR_TOKENS = OrderedDict(
    {
        "F": "Floor",
     }
)


SPECIAL_TOKENS = OrderedDict(
    {
        "B": "Block",
        "P": "Element (Lava, Water)",
        "O": "Element + Floor (Lava/Block, Water/Block)",
        "I": "Element + Block",
        "S": "Stair",
     }
)

ENEMY_TOKENS = OrderedDict(
    {
        "M": "Monster",
     }
)

TOKEN_DOWNSAMPLING_HIERARCHY = [
    VOID_TOKENS,
    FLOOR_TOKENS,
    WALL_TOKENS,
    SPECIAL_TOKENS,
    ENEMY_TOKENS,
]

TOKENS = OrderedDict(
    {**VOID_TOKENS, **FLOOR_TOKENS, **WALL_TOKENS, **SPECIAL_TOKENS, **ENEMY_TOKENS}
)

TOKEN_GROUPS = [VOID_TOKENS, FLOOR_TOKENS, WALL_TOKENS, SPECIAL_TOKENS, ENEMY_TOKENS]

REPLACE_TOKENS = {}

