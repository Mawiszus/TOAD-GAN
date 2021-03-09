import os
import glob
from utils import save_pkl

if __name__ == '__main__':
    obj_folder = "/home/awiszus/ownCloud/MarioSinGAN/Mineways/Drehmahl"
    obj_list = glob.glob(os.path.join(obj_folder, '*.obj'))
    coord_dict = {}

    for i, obj_name in enumerate(obj_list):
        with open(obj_name, 'r') as f:
            for line in f.readlines():
                if line[:44] == "#   Non-empty selection location min to max:":
                    words = line.split(" ")
                    # need to remove last character because of "," (8 does not have a ",")
                    numbers = [int(words[9][:-1]), int(words[10][:-1]), int(words[11]),
                               int(words[13][:-1]), int(words[14][:-1]), int(words[15][:-1])]
                    # coords formatting
                    coords = ((numbers[0], numbers[3]), (numbers[1], numbers[4]), (numbers[2], numbers[5]))

                    coord_dict[os.path.split(obj_name)[1][:-4]] = coords
                    print(os.path.split(obj_name)[1][:-4], coords)

    save_pkl(coord_dict, 'primordial_coords_dict', '../input/minecraft/')
