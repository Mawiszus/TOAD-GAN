
# for file in ../../output/real/objects/objects/*.obj
for file in ../../output/wandb/latest-run/files/objects/0/*.obj
do
  ~/bin/blender-2.79b-linux-glibc219-i686/blender -b --python ./CyclesMineways.py -- "${file}" 10.5 1
done