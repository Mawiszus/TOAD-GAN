from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.

VOID_TOKENS = OrderedDict(
    {
        "@": "Void",
     }
)

FLOOR_TOKENS = OrderedDict(
    {
        "-": "empty",
        "#": "solid",
     }
)

PLATFORM_TOKENS = OrderedDict(
    {
        "B": "Breakable",
        "M": "Moving Platform",
        "|": "Ladder",
        "t": "section of tiles that look solid but you fall through",
        "D": "Door",
     }
)

HAZARD_TOKENS = OrderedDict(
    {
        "H": "Hazard (spikes)",
        "C": "Cannon/Shooter",
    }
)

SPECIAL_TOKENS = OrderedDict(
    {
        "L": "Large Health Pack",
        "l": "Small Health Pack",
        "W": "Large Ammo Pack",
        "w": "Small Ammo Pack",
        "+": "Extra Life",
        "P": "Player",
        "U": "Transport Beam Upgrade (single appearance upgrade)",
        "*": "special item that completely fills health and ammo (only shows up in the final level)",
     }
)

TOKEN_DOWNSAMPLING_HIERARCHY = [
    VOID_TOKENS,
    FLOOR_TOKENS,
    PLATFORM_TOKENS,
    HAZARD_TOKENS,
    SPECIAL_TOKENS,
]

TOKENS = OrderedDict(
    {**VOID_TOKENS, **FLOOR_TOKENS, **PLATFORM_TOKENS, **HAZARD_TOKENS, **SPECIAL_TOKENS}
)

TOKEN_GROUPS = [VOID_TOKENS, FLOOR_TOKENS, PLATFORM_TOKENS, HAZARD_TOKENS, SPECIAL_TOKENS]

REPLACE_TOKENS = {"P": "-"}

