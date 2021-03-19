# Blender Scripts

This folder contains the scripts used in blender to make minecraft renders


**IMPORTANT:** CyclesMineways.py does not work with the newer Blender Versions. 
It was tested with Blender 2.79 and seems to be working there.

Usage:
```
& 'C:\Program Files\Blender Foundation\Blender_2_79\blender.exe' -b --python .\CyclesMineways.py -- "..\..\output\obj_test\2_2\0.obj" 8.5
```

``--`` is important, as it tells blender not to parse the following arguments and only the script parses them!

``-b`` is background