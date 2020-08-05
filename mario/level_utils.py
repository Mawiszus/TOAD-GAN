import torch
from loguru import logger

from .tokens import TOKEN_GROUPS, REPLACE_TOKENS


# Miscellaneous functions to deal with ascii-token-based levels.


def group_to_token(tensor, tokens, token_groups=TOKEN_GROUPS):
    """ Converts a token group level tensor back to a full token level tensor. """
    new_tensor = torch.zeros(tensor.shape[0], len(tokens), *tensor.shape[2:]).to(
        tensor.device
    )
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, i] = tensor[:, group_idx]
                break
    return new_tensor


def token_to_group(tensor, tokens, token_groups=TOKEN_GROUPS):
    """ Converts a full token tensor to a token group tensor. """
    new_tensor = torch.zeros(tensor.shape[0], len(token_groups), *tensor.shape[2:]).to(
        tensor.device
    )
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, group_idx] += tensor[:, i]
                break
    return new_tensor


def load_level_from_text(path_to_level_txt, replace_tokens=REPLACE_TOKENS):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level


def ascii_to_one_hot_level(level, tokens):
    """ Converts an ascii level to a full token level tensor. """
    oh_level = torch.zeros((len(tokens), len(level), len(level[-1])))
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in tokens and token != "\n":
                oh_level[tokens.index(token), i, j] = 1
    return oh_level


def one_hot_to_ascii_level(level, tokens):
    """ Converts a full token level tensor to an ascii level. """
    ascii_level = []
    for i in range(level.shape[2]):
        line = ""
        for j in range(level.shape[3]):
            line += tokens[level[:, :, i, j].argmax()]
        if i < level.shape[2] - 1:
            line += "\n"
        ascii_level.append(line)
    return ascii_level


def read_level(opt, tokens=None, replace_tokens=REPLACE_TOKENS):
    """ Wrapper function for read_level_from_file using namespace opt. Updates parameters for opt."""
    level, uniques = read_level_from_file(opt.input_dir, opt.input_name, tokens, replace_tokens)
    opt.token_list = uniques
    logger.info("Tokens in level {}", opt.token_list)
    opt.nc_current = len(uniques)
    return level


def read_level_from_file(input_dir, input_name, tokens=None, replace_tokens=REPLACE_TOKENS):
    """ Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
    Token. """
    txt_level = load_level_from_text("%s/%s" % (input_dir, input_name), replace_tokens)
    uniques = set()
    for line in txt_level:
        for token in line:
            # if token != "\n" and token != "M" and token != "F":
            if token != "\n" and token not in replace_tokens.items():
                uniques.add(token)
    uniques = list(uniques)
    uniques.sort()  # necessary! otherwise we won't know the token order later
    oh_level = ascii_to_one_hot_level(txt_level, uniques if tokens is None else tokens)
    return oh_level.unsqueeze(dim=0), uniques


def place_a_mario_token(level):
    """ Finds the first plausible spot to place Mario on. Especially important for levels with floating platforms.
    level is expected to be ascii."""
    # First check if default spot is available
    for j in range(1, 4):
        if level[-3][j] == '-' and level[-2][j] in ['X', '#', 'S', '%', 't', '?', '@', '!', 'C', 'D', 'U', 'L']:
            tmp_slice = list(level[-3])
            tmp_slice[j] = 'M'
            level[-3] = "".join(tmp_slice)
            return level

    # If not, check for first possible location from left
    for j in range(len(level[-1])):
        for i in range(1, len(level)):
            if level[i - 1][j] == '-' and level[i][j] in ['X', '#', 'S', '%', 't', '?', '@', '!', 'C', 'D', 'U', 'L']:
                tmp_slice = list(level[i - 1])
                tmp_slice[j] = 'M'
                level[i - 1] = "".join(tmp_slice)
                return level

    return level  # Will only be reached if there is no place to put Mario
