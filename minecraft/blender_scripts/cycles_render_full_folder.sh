for file in $1/*.obj
do
  ~/bin/blender-2.79b-linux-glibc219-i686/blender -b --python ./minecraft/blender_scripts/CyclesMineways.py -- "${file}" 14.5 0
done
mkdir -p $1/../renders
for file in $1/*render-0.png
do
  cp -rf "${file}" $1/../renders
done