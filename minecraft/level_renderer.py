import os
import subprocess
import wandb


def make_render_script(scriptpath, scriptname, obj_path, obj_name, worldname, coords):
    with open(os.path.join(scriptpath, scriptname) + '.mwscript', 'w') as f:
        f.write('Save Log file: ' + os.path.join(scriptpath, scriptname) + '.log\n')
        f.write('Set render type: Wavefront OBJ absolute indices\n')
        f.write('Minecraft world: ' + worldname + '\n')
        f.write('Selection location min to max: {}, {}, {} to {}, {}, {}\n'.format(
            coords[0][0], coords[1][0], coords[2][0],
            coords[0][1] - 1, coords[1][1] - 1, coords[2][1] - 1
        ))
        f.write("Scale model by making each block 100 cm high\n")
        # f.write("Scale model by fitting to a height of 100 cm\n")
        f.write('Export for Rendering: ' + os.path.join(obj_path, obj_name) + '.obj')


def make_obj(scriptpath, scriptnames, worldpath="../minecraft_worlds/"):
    commands = ['wine', 'minecraft/mineways/Mineways32.exe', '-m', '-s', worldpath]
    for name in scriptnames:
        commands.append(os.path.join(scriptpath, name) + '.mwscript')

    process = subprocess.Popen(commands,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)

    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)


def render_minecraft(world_name, coords_to_read, obj_path, obj_name):
    """Render a Minecraft snippet with Mineways. Returns the path to the rendered .obj file."""
    # os.makedirs("%s/objects/%s" % (obj_path, folder), exist_ok=True)
    # objectpath = os.path.join(basepath, "objects/" + folder + "/")
    make_render_script("minecraft/mineways/", obj_name, obj_path, obj_name, world_name, coords_to_read)
    make_obj("minecraft/mineways/", [obj_name, "close"])
    rendered_path = os.path.join(obj_path, obj_name + ".obj")
    return rendered_path
