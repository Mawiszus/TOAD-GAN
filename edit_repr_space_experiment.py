from utils import load_pkl, save_pkl

# Define lists to reset
l_dirt = ["minecraft:dirt", "minecraft:grass_block", "minecraft:farmland"]
l_ruin = ["minecraft:cobblestone", "minecraft:mossy_cobblestone", "minecraft:stone_bricks"]
l_cr_ruin = ["minecraft:cracked_stone_bricks"]
l_mo_ruin = ["minecraft:mossy_stone_bricks"]
l_slab = ["minecraft:stone_brick_slab"]
l_stairs = ["minecraft:stone_brick_stairs"]
l_tallgrass = ["minecraft:tall_grass"]
l_grass = ["minecraft:grass", "minecraft:large_fern", "minecraft:vine", "minecraft:wheat",
           "minecraft:oak_leaves", "minecraft:fern"]


def set_name(token):
    if token in l_dirt:
        n = "minecraft:sand"
    elif token in l_ruin:
        n = "minecraft:cut_red_sandstone"
    elif token in l_cr_ruin:
        n = "minecraft:chiseled_red_sandstone"
    elif token in l_mo_ruin:
        n = "minecraft:red_sandstone"
    elif token in l_slab:
        n = "minecraft:cut_red_sandstone_slab"
    elif token in l_stairs:
        n = "minecraft:smooth_red_sandstone_stairs"
    elif token in l_tallgrass:
        n = "minecraft:dead_bush"
    elif token in l_grass:
        n = "minecraft:air"
    else:
        n = token
    return n


def adjust_token_list(token_list):
    new_list = []
    for t in token_list:
        new_list.append(set_name(t))

    return new_list


if __name__ == '__main__':
    # Load repr space
    # block2repr = load_pkl('representations', prepath='/home/awiszus/Project/TOAD-GAN/output/block2vec/')
    block2repr = load_pkl("representations",
                          f"/home/schubert/projects/TOAD-GAN/input/minecraft/ruins/")

    # Edit repr space
    new_repr = {}
    for tok in list(block2repr.keys()):
        name = set_name(tok)
        if name in new_repr.keys():
            new_repr[name].append(block2repr[tok])
        else:
            new_repr[name] = [block2repr[tok]]
        print("Set {} to {}".format(tok, name))
        # print(block2repr[tok], new_repr[name])

    # Save new repr space
    # save_pkl(new_repr, 'edited_representations', prepath='/home/awiszus/Project/TOAD-GAN/output/block2vec/')
    save_pkl(new_repr, 'edited_representations', prepath='/home/awiszus/Project/TOAD-GAN/input/minecraft/')

    # Generate samples with new repr space (use generate_minecraft_samples.py)
    print("Done!")
