# Blender Scripts

This folder contains the scripts used in blender to make minecraft renders


**IMPORTANT:** CyclesMineways.py does not work with the newer Blender Versions. 
It was tested with Blender 2.79 and seems to be working there.

Usage:

```
~/bin/blender-2.79b-linux-glibc219-i686/blender -b --python minecraft/blender_scripts/CyclesMineways.py -- output/obj_test/2/1.obj 8.5 0

& 'C:\Program Files\Blender Foundation\Blender_2_79\blender.exe' -b --python .\minecraft\blender_scripts\CyclesMineways.py -- "output\obj_test\2_2\0.obj" 8.5 0
```

``--`` is important, as it tells blender not to parse the following arguments and only the script parses them!

``-b`` is background

The arguments after ``--`` are:
1. path to ``.obj`` 
2. orthogonal scale (zoom, default: 8.5)
3. which view preset to use (available: 0,1,2,3)

The ``.png`` will be saved in the same directory as the .obj file.