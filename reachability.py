import numpy as np
import os
import glob

from mario.level_image_gen import LevelImageGen

horizontal = 10  # when sprinting Mario can jump over a 10 tiles gap horizontally
vertical = 4  # Mario can jump over a 4 tile wall
diagonal = 6  # Mario can jump over 6 tiles to clear a 4 block height difference when sprinting

empty = '-'
ignored = ['M', 'F', '|', 'E', 'g', 'k', 'r', 'y', 'G', 'K', 'R', 'Y', '*', 'B', 'o']


def remove_ignored(level):
    """
    Replaces all ignored tokens with the empty token in a level. In case of Mario and the flag the coordinates of the blocks below are returned and they are also replaced.
    :param level: a level in ASCII form
    :return: the level in ASCII form with the ignored tokens replaced and the coordinates of the block below Mario and the flag if existing
    """
    new_level = []
    mario = (-1, -1)
    flag = (-1, -1)

    for i, row in enumerate(level):
        mario_y = row.find('M')
        if mario_y >= 0:
            mario = (i + 1, mario_y)
        flag_y = row.find('F')
        if flag_y >= 0:
            flag = (i + 1, flag_y)
        for token in ignored:
            row = row.replace(token, empty)
        new_level.append(row)

    return new_level, mario, flag


def reachability_map(level, shape, has_mario=False, has_flag=False, check_outside=False):
    """
    This creates a numpy 2D array containing the reachability map for a given ASCII-Level.
    Every solid block will have a 1 if Mario can stand on it and can reach the tile and a 0 else.
    Currently ignoring sprint.
    Levels are generated without Mario and the flag and as such the algorithm is not including these.
    :param level: The level (slice) as a list containing the ASCII strings of each level row
    :param shape: The level shape
    :param has_mario: As levels are expected to be generated without Mario, this option has to be set to search for Mario as a starting point
    :param has_flag: As levels are expected to be generated without the flag, this option has to be set to determine playability via reaching the flag
    :param check_outside: If this option is set, playability will check if the player can reach the outside
    :return: A numpy array where a 0 indicates an unreachable block and a 1 denotes a reachable block; a boolean indicating if the level can be finished by the player
    """

    level, mario, flag = remove_ignored(level)
    map = np.zeros(shape=shape)
    index_queue = []

    # find the starting point, either the block Mario is standing on or the first solid block Mario could stand on
    found_first = False
    if has_mario:
        index_queue.append(mario)
    else:
        for i in range(shape[0] - 1, 0, -1):  # start from the bottom of the level
            for j in range(0, shape[1]):
                tile = level[i][j]
                if tile != empty and (map[i][j] == 1 or not found_first and i < shape[0] - 1) and i > 0 and \
                        level[i - 1][j] == empty:
                    found, queue, _ = mark(level, map, i, j)
                    index_queue.extend(queue)
                    if not found_first:
                        found_first = found
                        break
            if found_first:
                break

    # calculate all reachable positions by applying a BFS type of algorithm
    outside = False
    while len(index_queue) > 0:
        index = index_queue.pop()
        _, queue, reached_outside = mark(level, map, index[0], index[1], check_outside=check_outside)
        if reached_outside:
            outside = True
        index_queue.extend(queue)

    # a level is playable if either the flag is reachable or if no flag is included, the rightmost side can be reached
    # Bug: if the level ends with a gap, it might be playable but still wouldn't count as such
    playable = False

    if has_flag:
        if map[flag[0]][flag[1]]:
            playable = True
    else:
        # look at all tiles in the last column
        for i in range(1, shape[0]):
            if map[shape[0] - i][shape[1] - 1]:
                playable = True
                break

        if not playable and check_outside:
            # Assumption is that reaching the outside is identical to completing the level
            if outside:
                playable = True

    return map, playable


def check_blocked(level, i, j, dh, dv, right):
    """
    Checks for a given position, level and direction if a blockade exists in the range specified by dh and dv.
    :param level: The level in ASCII form
    :param i: x coordinate of the starting position
    :param j: y coordinate of the starting position
    :param dh: amount of blocks in the horizontal direction from the starting point the algorithm tries to jump
    :param dv: amount of blocks in the vertical direction from the starting point the algorithm tries to jump
    :param right: direction of the jump
    :return: the blockade y value if a blockade is found, default max value otherwise
    """
    blocked = horizontal + 1  # default value
    boundary = j + dh if right else j - dh
    try:
        if level[i - dv][boundary] != empty:
            height = 1 + dv
            while height < vertical + 1:
                v = i - dv - height
                if v < 0:
                    # over maximum level height, cannot pass
                    blocked = dh
                    break
                if level[v][boundary] != empty or dh + height > 10:
                    height += 1
                else:
                    break
            if height == vertical + 1:
                blocked = dh
    except IndexError:
        # over maximum level height, cannot pass
        blocked = dh

    return blocked


def check_down(level, map, i, j, dh, check_outside, right):
    drop = 1
    found_first = False
    reach_outside = False
    found = []
    boundary = j + dh if right else j - dh
    if boundary > map.shape[1] - 1:
        if check_outside:
            reach_outside = True
    else:
        y = min(max(boundary, 0), map.shape[1] - 1)
        while i + drop < map.shape[0]:
            # right and down
            x = i + drop
            above = x - 1

            if level[x][y] != empty and above >= 0 and level[above][y] == empty and map[x][y] != 1:
                map[x][y] = 1
                found.append((x, y))
                found_first = True
                break
            drop += 1

    return found_first, reach_outside, found


def mark(level, map, i, j, check_outside=False):
    """
    For a given position and a level this will mark all tiles reachable from the given position and collect all these positions for further use.
    :param level: The level (slice) as a list containing the ASCII strings of each level row
    :param map: The current reachability map where the reachable tiles will be marked
    :param i: x coordinate
    :param j: y coordinate
    :param check_outside: if the algorithm should indicate that the player can reach the right outside of the level
    :return: A boolean indicating if any tile can be reached from this position, a list of all reachable positions and if the outside can be reached
    """
    found_first = False
    reach_outside = False
    found = []
    blocked_level = vertical + 1
    blocked_right = horizontal + 1
    blocked_left = horizontal + 1
    blocked_down_right = horizontal + 1
    blocked_down_left = horizontal + 1

    # mark diagonally
    for dh in range(0, horizontal + 1):
        # check down as far as possible, Mario can fall down the whole level until he hits a solid block
        if blocked_down_right == horizontal + 1:
            blocked_down_right = check_blocked(level, i, j, dh, 0, right=True)
        if blocked_down_right >= dh:
            found_rechable, found_outside, positions = check_down(level, map, i, j, dh, check_outside, right=True)
            if found_rechable:
                found_first = True
            if found_outside:
                reach_outside = True
            found.extend(positions)

        if blocked_down_left == horizontal + 1:
            blocked_down_left = check_blocked(level, i, j, dh, 0, right=False)
        if blocked_down_left >= dh:
            found_rechable, found_outside, positions = check_down(level, map, i, j, dh, check_outside, right=False)
            if found_rechable:
                found_first = True
            if found_outside:
                reach_outside = True
            found.extend(positions)

        for dv in range(0, vertical + 1):
            if dh != 0 or dv != 0:
                if dv >= blocked_level:
                    break

                # check if vertical path is blocked
                if dh == 0:
                    if level[i - dv][j] != empty:
                        blocked_level = dv
                        continue

                # check if horizontal right path is blocked
                if blocked_right == horizontal + 1:
                    blocked_right = check_blocked(level, i, j, dh, dv, right=True)

                if dh <= blocked_right and dh + dv <= 10:
                    # right and up
                    x = min(max(i - dv, 0), map.shape[0] - 1)
                    right = j + dh
                    if right > map.shape[1] - 1 and check_outside:
                        reach_outside = True
                    y = min(max(right, 0), map.shape[1] - 1)
                    above = x - 1
                    if level[x][y] != empty and above >= 0 and level[above][y] == empty and map[x][y] != 1:
                        map[x][y] = 1
                        found.append((x, y))
                        found_first = True

                # check if horizontal left path is blocked
                if blocked_left == horizontal + 1:
                    blocked_left = check_blocked(level, i, j, dh, dv, right=False)

                if dh <= blocked_left and dh + dv <= 10:
                    # left and up
                    x = min(max(i - dv, 0), map.shape[0] - 1)
                    y = min(max(j - dh, 0), map.shape[1] - 1)
                    above = x - 1
                    if level[x][y] != empty and above >= 0 and level[above][y] == empty and map[x][y] != 1:
                        map[x][y] = 1
                        found.append((x, y))
                        found_first = True

    return found_first, found, reach_outside


def load_level(path):
    """
    Reads in a level from a given path (with the complete file name). This might be doubling another already existing method.
    :param path: Full path to the level
    :return: The level as an array with each entry being a String of one level row in ASCII form
    """
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            lines.append(line.strip())

    return lines


def calc_percentages(level, map):
    """
    Calculates various statistics for a given level and its corresponding reachability map.
    :param level: The level (slice) as a list containing the ASCII strings of each level row
    :param map: The reachability map for the level
    :return: A dictionary containing stats: total blocks, solid blocks and solid block percentage, reachable blocks and reachable blocks percentage, accessible percentage
    """
    stats = {
        'total': 0,
        'ignored': 0,
        'solid': 0,
        'reachable': 0,
        'non_accessible': 0
    }
    for i, line in enumerate(level):
        if i != 15:
            for j, char in enumerate(line):
                stats['total'] += 1
                if char in ignored or char == empty:
                    stats['ignored'] += 1
                else:
                    stats['solid'] += 1
                    # check if this block is below another solid block, e.g. not accessible
                    above = level[i - 1][j]
                    if map[i][j] == 0 and i > 0 and above not in ignored and above != empty:
                        stats['non_accessible'] += 1
                if map[i][j] == 1:
                    stats['reachable'] += 1

    stats['solid_perc'] = stats['solid'] / stats['total']
    stats['reachable_perc'] = stats['reachable'] / stats['solid']
    stats['accessible_reachable_perc'] = stats['reachable'] / (stats['solid'] - stats['non_accessible'])

    return stats


def test_generated(path, playable_only=False, has_mario=False, has_flag=False, check_outside=False):
    """
    Caluclates the stats and reachability maps for all levels saved at the given path. The stats are printed on the console.
    :param path: String for the path to the folder containing all level folders
    :param playable_only:
    :return: This method does not return anything
    """
    total = 0
    solid_perc = 0
    reachable_perc = 0
    accessible_perc = 0

    # print table header
    if playable_only:
        print('Level & solid perc & reachable perc & accessible perc \\\ ')
    else:
        print('Level & solid perc & reachable perc & accessible perc & playable perc \\\ ')

    for i, baseline_level_dir in enumerate(sorted(os.listdir(path))):
        count = 0
        data = []
        solids = []
        reachables = []
        accessibles = []
        playable_count = 0
        for level in glob.glob(path + baseline_level_dir + '/*.txt'):
            l = load_level(level)
            map, playable = reachability_map(l, (16, len(l[0])), has_mario=has_mario, has_flag=has_flag, check_outside=check_outside)

            if playable_only:
                if playable:
                    stats = calc_percentages(l, map)
                    data.append(stats['reachable_perc'])
                    solids.append(stats['solid_perc'])
                    reachables.append(stats['reachable_perc'])
                    accessibles.append(stats['accessible_reachable_perc'])
                    count += 1
                    total += stats['total']
                    solid_perc += stats['solid_perc']
                    reachable_perc += stats['reachable_perc']
                    accessible_perc += stats['accessible_reachable_perc']

            else:
                if playable:
                    playable_count += 1
                else:
                    save_image(image_gen, l, map, './output/{0}_unplayable_{1}.png'.format(baseline_level_dir, playable_count))

                stats = calc_percentages(l, map)
                data.append(stats['reachable_perc'])
                solids.append(stats['solid_perc'])
                reachables.append(stats['reachable_perc'])
                accessibles.append(stats['accessible_reachable_perc'])
                count += 1
                total += stats['total']
                solid_perc += stats['solid_perc']
                reachable_perc += stats['reachable_perc']
                accessible_perc += stats['accessible_reachable_perc']

        total /= count
        solid_perc /= count
        reachable_perc /= count
        accessible_perc /= count
        playable_count /= count

        cur_level = baseline_level_dir.split('_')[1]
        if playable_only:
            print('G {0} & {1:.2f} & {2:.2f} & {3:.2f} \\\ '.format(cur_level, solid_perc,
                                                                          reachable_perc, accessible_perc))
        else:
            print('G {0} & {1:.2f} & {2:.2f} & {3:.2f} & {4} \\\ '.format(cur_level, solid_perc,
                                                                      reachable_perc, accessible_perc,
                                                                      playable_count))
        total = 0
        solid_perc = 0
        reachable_perc = 0
        accessible_perc = 0


def save_image(image_gen, l, map, path):
    """
    Saves an image of a given level with the reachable blocks marked at the specified path.
    :param image_gen: image_gen object needed for rendering
    :param l: the level
    :param map: the reachability map for the given level
    :param path: the full save path
    :return: This method does not return anything
    """

    # levels would most probably be in correct form already
    ascii_level = []
    for line in l:
        ascii_level.append(''.join(line))
    sample_image = image_gen.render(ascii_level, map)
    sample_image.save(path)


if __name__ == '__main__':
    test_generated_levels = True
    test = False
    image_gen = LevelImageGen('./mario/sprites/')
    if test_generated_levels:
        path = './input/baselines/'
        test_generated(path, check_outside=True)
        test_generated(path, playable_only=True, check_outside=True)
    elif test:
        path = './input/test/test_outside.txt'
        l = load_level(path)
        map, playable = reachability_map(l, (16, len(l[0])), check_outside=True, has_mario=False, has_flag=False)
        print(playable)
        save_image(image_gen, l, map, './output/test_outside.png')
    else:
        # test original levels
        for path in glob.glob('./input' + '/*.txt'):
            level = load_level(path)
            map, _ = reachability_map(level, (16, len(level[0])), has_mario=True, has_flag=True)
            stats = calc_percentages(level, map)
            level_name = path.split('/')[-1].split('.')[0].split('_')[-1]
            print('{0} & {1:.2f} & {2:.2f} & {3:.2f} \\\ '.format(level_name, stats['solid_perc'], stats['reachable_perc'],
                                                                  stats['accessible_reachable_perc']))
            path = './output/orig_{0}_reachability.png'.format(level_name)
            save_image(image_gen, level, map, path)
