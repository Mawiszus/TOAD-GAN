import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Add level files.")
    parser.add_argument('--levels', type=str, nargs='+', required=True, help="Level File names to concatenate")
    args = parser.parse_args()

    levelnames = [os.path.basename(x)[:-4] for x in args.levels]
    concat_file = open(os.path.join(os.path.dirname(args.levels[0]), ''.join(levelnames) + '.txt'), "w")
    files = []
    for level_name in args.levels:
        files.append(open(level_name, 'r'))

    for line in files[0]:
        currlines = [line.replace('\n', '')]
        for file in files[1:]:
            currline = file.readline().replace('\n', '')
            currlines.append(currline)
        concat_file.writelines(''.join(currlines) + '\n')

    concat_file.close()


