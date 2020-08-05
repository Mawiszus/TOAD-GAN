from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.

GROUND_TOKENS = OrderedDict(
    {"X": "Ground Block"}
)

SPECIAL_GROUND_TOKENS = OrderedDict(
    {
        "#": "Pyramind Block",
    }
)

PLATFORM_TOKENS = OrderedDict(
    {
        "S": "Normal Brick Block",
        "%": "Jump through platform",
        "|": "Background for the jump through platform",
        "b": "Bullet bill neck or body",
    }
)

SKY_TOKENS = OrderedDict(
    {
        "-": "Empty",
    }
)

PIPE_TOKENS = OrderedDict(
    {
        "<": "Top left of empty pipe",
        ">": "Top right of empty pipe",
        "[": "Left of empty pipe",
        "]": "Right of empty pipe",
        "t": "Empty Pipe",
        "T": "Pipe with Piranaha Plant in it",
    }
)

ENEMY_TOKENS = OrderedDict(
    {
        "E": "Goomba",
        "g": "Goomba",
        "k": "Green Koopa",
        "r": "Red Koopa",
        "y": "Spiky",
    }
)

SPECIAL_ENEMY_TOKENS = OrderedDict(
    {
        "G": "Winged Goomba",
        "K": "Winged Green Koopa",
        "R": "Winged Red Koopa",
        "Y": "Winged Spiky",
        "*": "Bullet Bill",
        "B": "Bullet bill head",
    }
)

SPECIAL_TOKENS = OrderedDict(
    {
        "o": "Coin",
        "Q": "Coin Question block",
        "!": "Coin Question block",
        "?": "Special Question block",
        "@": "Special Question block",
        "M": "Mario Starting Position, not having it will force the engine to start at x = 0 and the first ground floor.",
        "F": "Mario finish line, not having it will force the engine to end at x = levelWidth and the first ground floor.",
        "C": "Coing Brick Block",

    }
)

EXTRA_SPECIAL_TOKENS = OrderedDict(
    {
        "D": "Used block",
        "U": "Musrhoom Brick Block",
        "L": "1 up Block",
        "2": "Invisible coin bock",
        "1": "Invisible 1 up block",
    }
)

TOKEN_DOWNSAMPLING_HIERARCHY = [
    SKY_TOKENS,
    GROUND_TOKENS,
    SPECIAL_GROUND_TOKENS,
    PLATFORM_TOKENS,
    PIPE_TOKENS,
    ENEMY_TOKENS,
    SPECIAL_ENEMY_TOKENS,
    SPECIAL_TOKENS,
    EXTRA_SPECIAL_TOKENS
]


TOKENS = OrderedDict(
    {**GROUND_TOKENS, **SPECIAL_GROUND_TOKENS, **PLATFORM_TOKENS, **SKY_TOKENS, **PIPE_TOKENS,
     **ENEMY_TOKENS, **SPECIAL_ENEMY_TOKENS, **SPECIAL_TOKENS, **EXTRA_SPECIAL_TOKENS}
)

TOKEN_GROUPS = [SKY_TOKENS, GROUND_TOKENS, SPECIAL_GROUND_TOKENS, PLATFORM_TOKENS, ENEMY_TOKENS,
                SPECIAL_ENEMY_TOKENS, PIPE_TOKENS, SPECIAL_TOKENS, EXTRA_SPECIAL_TOKENS]

REPLACE_TOKENS = {"F": "-", "M": "-"}  # We replace these tokens so the generator doesn't add random start or end points

