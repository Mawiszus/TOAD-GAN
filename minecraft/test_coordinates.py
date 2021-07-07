from tqdm import tqdm
from utils import load_pkl
from minecraft.level_utils import read_level_from_file, save_oh_to_wrld_directly, clear_empty_world
from minecraft.level_renderer import render_minecraft


def get_coords(coord_dict, input_area_name, sub_coords):
    tmp_coords = coord_dict[input_area_name]
    sub_coords = [(sub_coords[0], sub_coords[1]),
                  (sub_coords[2], sub_coords[3]),
                  (sub_coords[4], sub_coords[5])]
    coords = []
    for i, (start, end) in enumerate(sub_coords):
        curr_len = tmp_coords[i][1] - tmp_coords[i][0]
        if isinstance(start, float):
            tmp_start = curr_len * start + tmp_coords[i][0]
            tmp_end = curr_len * end + tmp_coords[i][0]
        elif isinstance(start, int):
            tmp_start = tmp_coords[i][0] + start
            tmp_end = tmp_coords[i][0] + end
        else:
            AttributeError("Unexpected type for sub_coords")
            tmp_start = tmp_coords[i][0]
            tmp_end = tmp_coords[i][1]

        coords.append((int(tmp_start), int(tmp_end)))
    return coords


if __name__ == '__main__':
    coord_dictionary = load_pkl('primordial_coords_dict', 'input/minecraft/')
    clear_empty_world('/home/awiszus/Project/minecraft_worlds/', 'Gen_Empty_World')

    sub_coord_dictionary = {
        "ruins": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "simple_beach": [0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
        "desert": [0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
        "plains": [0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
        "swamp": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "vanilla_village": [0.33333, 0.66667, 0.0, 1.0, 0.33333, 0.66667],
        "vanilla_mineshaft": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        # "bay_structure": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    }

    for n, key in tqdm(enumerate(sub_coord_dictionary.keys())):
        coords = get_coords(coord_dictionary, key, sub_coord_dictionary[key])
        level, uniques, props = read_level_from_file("/home/awiszus/Project/minecraft_worlds/",
                                                     "Drehmal v2.1 PRIMORDIAL", coords, None, None)
        tmp_coords = (0, 0, 0)
        save_oh_to_wrld_directly('/home/awiszus/Project/minecraft_worlds/', 'Gen_Empty_World', tmp_coords, level,
                                 None, None, uniques, props)
        curr_coords = [[0, level.shape[-3]], [0, level.shape[-2]], [0, level.shape[-1]]]
        render_minecraft('Gen_Empty_World', curr_coords, "output/real/objects", key)
        print(key, " : ", level.shape)
