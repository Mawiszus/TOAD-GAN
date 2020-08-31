import os

from PIL import Image, ImageOps, ImageEnhance


class LevelImageGen:
    """ Generates PIL Image files from The Legend of Zelda ascii levels.
    Initialize once and then use LevelImageGen.render() to generate images. """
    def __init__(self, sprite_path):
        """ sprite_path: path to the folder of sprite files, e.g. 'zelda/sprites/' """

        # Load Graphics
        mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))

        # Cut out the actual sprites:
        sprite_dict = dict()

        # Map Sheet TODO: update generator and mapsheet to deal with walls
        map_names = ['culul', 'culur', 'curul', 'curur', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                     '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                     '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                     '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                     '-', '-', '-', '-', 'dcbul', 'dcbur', 'dobul', 'dobur', 'B', 'F', 'S', 'P', '-', '-', '-', '-',
                     '-', '-', '-', '-', 'dcbbl', 'dcbbr', 'dobbl', 'dobbr', 'Ml', 'Mr', '-', 'I', '-', '-', '-', '-',
                     '-', '-', '-', '-', 'dcuul', 'dcuur', 'douul', 'douur', 'MIl', 'MIr', '-', 'O', '-', '-', '-', '-',
                     '-', '-', '-', '-', 'dcubl', 'dcubr', 'doubl', 'doubr', '-', '-', '-', '-', '-', '-', '-', '-',
                     '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                     '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                     '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                     ]

        sprite_dict['-'] = mapsheet.crop((15*16, 10*16, 16*16, 11*16))
        sprite_dict['B'] = mapsheet.crop((8*16, 4*16, 9*16, 5*16))
        sprite_dict['F'] = mapsheet.crop((9*16, 4*16, 10*16, 5*16))
        sprite_dict['S'] = mapsheet.crop((10*16, 4*16, 11*16, 5*16))
        sprite_dict['P'] = mapsheet.crop((11*16, 4*16, 12*16, 5*16))
        sprite_dict['Ml'] = mapsheet.crop((8*16, 5*16, 9*16, 6*16))
        sprite_dict['Mr'] = mapsheet.crop((9*16, 5*16, 10*16, 6*16))
        sprite_dict['I'] = mapsheet.crop((11*16, 5*16, 12*16, 6*16))
        sprite_dict['O'] = mapsheet.crop((11*16, 6*16, 12*16, 7*16))

        sprite_dict['W'] = mapsheet.crop((7*16, 5*16 + 8, 8*16, 6*16 + 8))
        sprite_dict['Dd'] = mapsheet.crop((6*16 + 8, 4*16, 7*16 + 8, 5*16))
        sprite_dict['Du'] = mapsheet.crop((6*16 + 8, 7*16, 7*16 + 8, 8*16))
        sprite_dict['Dr'] = mapsheet.crop((8*16, 9*16, 9*16, 10*16))
        sprite_dict['Dl'] = mapsheet.crop((11*16, 9*16, 12*16, 10*16))

        self.sprite_dict = sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """ Helper to make correct sprites and sprite sizes to draw into the image.
         Some sprites are bigger than one tile and the renderer needs to adjust for them."""

        # Init default size
        new_left = curr_x * 16
        new_top = curr_y * 16
        new_right = (curr_x + 1) * 16
        new_bottom = (curr_y + 1) * 16

        # Handle sprites depending on their type:
        if sprite_key == 'M':

            actual_sprite = self.sprite_dict['Ml']

            # for i in range(curr_x, len(ascii_level)):
            for i in range(curr_x-1, -1, -1):
                if ascii_level[i][curr_y] in ['W', 'D']:
                    break
                elif ascii_level[i][curr_y] == 'M':
                    actual_sprite = self.sprite_dict['Mr']
                    break
        elif sprite_key == 'D':
            if curr_y-1 >= 0 and not ascii_level[curr_x][curr_y-1] in ['W', 'D']:
                actual_sprite = self.sprite_dict['Dd']
            elif curr_x-1 >= 0 and not ascii_level[curr_x-1][curr_y] in ['W', 'D']:
                actual_sprite = self.sprite_dict['Dr']
            elif curr_x+1 < len(ascii_level) and not ascii_level[curr_x+1][curr_y] in ['W', 'D']:
                actual_sprite = self.sprite_dict['Dl']
            else:
                actual_sprite = self.sprite_dict['Du']
        else:
            actual_sprite = self.sprite_dict[sprite_key]

        return actual_sprite, (new_left, new_top, new_right, new_bottom)

    def render(self, ascii_level):
        """ Renders the ascii level as a PIL Image. Assumes the Background is void """
        len_level = len(ascii_level)
        height_level = len(ascii_level[-1])

        # Fill base image with sky tiles
        dst = Image.new('RGB', (len_level*16, height_level*16))
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(self.sprite_dict['-'], (x*16, y*16, (x+1)*16, (y+1)*16))

        # Fill with actual tiles
        for y in range(height_level):
            for x in range(len_level):
                curr_sprite = ascii_level[x][y]
                sprite, box = self.prepare_sprite_and_box(ascii_level, curr_sprite, x, y)
                dst.paste(sprite, box, mask=sprite)

        return dst
