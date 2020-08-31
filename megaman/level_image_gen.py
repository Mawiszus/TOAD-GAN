import os

from PIL import Image, ImageOps, ImageEnhance


class LevelImageGen:
    """ Generates PIL Image files from Megaman ascii levels.
    Initialize once and then use LevelImageGen.render() to generate images. """
    def __init__(self, sprite_path, n_sheet=0):
        """ sprite_path: path to the folder of sprite files, e.g. 'megaman/sprites/' """

        # Load Graphics
        if n_sheet == 0:
            mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))
        elif n_sheet == 5:
            mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet_5.png'))
        else:
            mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))  # default map sheet is lvl 1

        # Cut out the actual sprites:
        sprite_dict = dict()

        # Map Sheet
        map_names = ['@', '#', 'sky', 'B', 'H', '|',
                     'L', 'l', 'W', 'w', '+', 'M',
                     'C', 'D', 'U', 't', '*', '-',
                     'M_e', '#_ul', '#_l', '#_m', '#_ur', '#_r',
                     ]

        sheet_length = (4, 6)
        sprite_counter = 0
        for i in range(sheet_length[0]):
            for j in range(sheet_length[1]):
                sprite_dict[map_names[sprite_counter]] = mapsheet.crop((j*16, i*16, (j+1)*16, (i+1)*16))
                sprite_counter += 1

        sprite_dict['P'] = mapsheet.crop((0, 64, 32, 96))

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
        if sprite_key == 'P':

            actual_sprite = self.sprite_dict['P']
            new_right += 16
            new_top -= 16
        if sprite_key == '#':
            if curr_x > 0 and ascii_level[curr_y][curr_x-1] not in ['#', '@', '\n']:
                if curr_x < len(ascii_level[curr_y])-1 and ascii_level[curr_y][curr_x+1] in ['#', '@', '\n']:
                    # left end
                    if curr_y > 0 and ascii_level[curr_y-1][curr_x] != '#':
                        # top left corner
                        actual_sprite = self.sprite_dict['#_ul']
                    else:
                        actual_sprite = self.sprite_dict['#_l']
                else:
                    actual_sprite = self.sprite_dict['#']
            elif curr_x < len(ascii_level[curr_y])-1 and ascii_level[curr_y][curr_x+1] not in ['#', '@', '\n']:
                if curr_x > 0 and ascii_level[curr_y][curr_x - 1] in ['#', '@', '\n']:
                    # right end
                    if curr_y > 0 and ascii_level[curr_y-1][curr_x] != '#':
                        # top right corner
                        actual_sprite = self.sprite_dict['#_ur']
                    else:
                        actual_sprite = self.sprite_dict['#_r']
                else:
                    actual_sprite = self.sprite_dict['#']
            elif curr_y > 0 and ascii_level[curr_y-1][curr_x] == '#':
                actual_sprite = self.sprite_dict['#_m']
            else:
                actual_sprite = self.sprite_dict['#']
        elif sprite_key == 'M':
            if curr_x > 0 and ascii_level[curr_y][curr_x-1] not in ['M', '@', '\n']:
                # left end
                actual_sprite = self.sprite_dict['M_e']
            elif curr_x < len(ascii_level[curr_y])-1 and ascii_level[curr_y][curr_x+1] not in ['M', '@', '\n']:
                # right end
                actual_sprite = self.sprite_dict['M_e']
            else:
                actual_sprite = self.sprite_dict['M']
        else:
            actual_sprite = self.sprite_dict[sprite_key]

        return actual_sprite, (new_left, new_top, new_right, new_bottom)

    def render(self, ascii_level):
        """ Renders the ascii level as a PIL Image. Assumes the Background is sky """
        len_level = len(ascii_level[-1])
        height_level = len(ascii_level)

        # Fill base image with sky tiles
        dst = Image.new('RGB', (len_level*16, height_level*16))
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(self.sprite_dict['sky'], (x*16, y*16, (x+1)*16, (y+1)*16))

        # Fill with actual tiles
        for y in range(height_level):
            for x in range(len_level):
                curr_sprite = ascii_level[y][x]
                sprite, box = self.prepare_sprite_and_box(ascii_level, curr_sprite, x, y)
                dst.paste(sprite, box, mask=sprite)

        return dst
