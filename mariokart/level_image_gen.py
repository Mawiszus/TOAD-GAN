import os

from PIL import Image, ImageOps, ImageEnhance


class LevelImageGen:
    """ Generates PIL Image files from Mario Kart ascii levels.
    Initialize once and then use LevelImageGen.render() to generate images. """
    def __init__(self, sprite_path):
        """ sprite_path: path to the folder of sprite files, e.g. 'mariokart/sprites/' """

        # Load Graphics
        mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))

        # Cut out the actual sprites:
        sprite_dict = dict()

        # Map Sheet
        map_names = ['W_r', 'W_g', 'W_y', 'W_b', 'W_0',
                     '-', 'R', 'S', 'C', 'empty',
                     'O_ul', 'O_ur', 'Q_ul', 'Q_ur', 'Q',
                     'O_dl', 'O_dr', 'Q_dl', 'Q_dr', 'O',
                     '<_ul', '<_ur', '<', 'empty', 'empty',
                     '<_dl', '<_dr', 'empty', 'empty', 'empty',
                     ]

        sheet_length = (6, 5)
        sprite_counter = 0
        for i in range(sheet_length[0]):
            for j in range(sheet_length[1]):
                sprite_dict[map_names[sprite_counter]] = mapsheet.crop((j*8, i*8, (j+1)*8, (i+1)*8))
                sprite_counter += 1

        self.sprite_dict = sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """ Helper to make correct sprites and sprite sizes to draw into the image.
         Some sprites are bigger than one tile and the renderer needs to adjust for them."""

        # Init default size
        new_left = curr_x * 8
        new_top = curr_y * 8
        new_right = (curr_x + 1) * 8
        new_bottom = (curr_y + 1) * 8

        # Handle sprites depending on their type:
        if sprite_key in ['O', 'Q', '<']:
            if curr_x > 0 and ascii_level[curr_y][curr_x-1] == sprite_key:
                if curr_y > 0 and ascii_level[curr_y-1][curr_x] == sprite_key:
                    if ascii_level[curr_y-1][curr_x-1] == sprite_key:
                        # 4 Sprites of the same type! use big sprite
                        new_left -= 8
                        new_top -= 8
                        actual_sprite = Image.new('RGBA', (2 * 8, 2 * 8))
                        actual_sprite.paste(self.sprite_dict[sprite_key + '_ul'], (0, 0, 8, 8))
                        actual_sprite.paste(self.sprite_dict[sprite_key + '_ur'], (8, 0, 2*8, 8))
                        actual_sprite.paste(self.sprite_dict[sprite_key + '_dl'], (0, 8, 8, 2*8))
                        actual_sprite.paste(self.sprite_dict[sprite_key + '_dr'], (8, 8, 2*8, 2*8))
                        return actual_sprite, (new_left, new_top, new_right, new_bottom)

            actual_sprite = self.sprite_dict[sprite_key]

        elif sprite_key == 'W':
            walls = [['W_r', 'W_g', 'W_y', 'W_b'],
                     ['W_g', 'W_y', 'W_b', 'W_r'],
                     ['W_y', 'W_b', 'W_r', 'W_g'],
                     ['W_b', 'W_r', 'W_g', 'W_y']]
            curr_col = curr_x % 16
            curr_row = curr_y % 16
            w_col = curr_col // 4
            w_row = curr_row // 4
            actual_sprite = self.sprite_dict[walls[w_col][w_row]]

        else:
            actual_sprite = self.sprite_dict[sprite_key]

        return actual_sprite, (new_left, new_top, new_right, new_bottom)

    def render(self, ascii_level):
        """ Renders the ascii level as a PIL Image. Assumes the Background is sky """
        len_level = len(ascii_level[-1])
        height_level = len(ascii_level)

        # Fill base image with sky tiles
        dst = Image.new('RGB', (len_level*8, height_level*8))
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(self.sprite_dict['-'], (x*8, y*8, (x+1)*8, (y+1)*8))

        # Fill with actual tiles
        for y in range(height_level):
            for x in range(len_level):
                curr_sprite = ascii_level[y][x]
                sprite, box = self.prepare_sprite_and_box(ascii_level, curr_sprite, x, y)
                dst.paste(sprite, box, mask=sprite)

        return dst
