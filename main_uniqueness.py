
import os
import copy
import numpy as np
from tap import Tap
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from mario.level_snippet_dataset import LevelSnippetDataset


class Config(Tap):
    baseline_level_dirs: str = "input/umap_images/baselines"
    slice_width: int = 16


def main():
    opt = Config().parse_args()
    token_list = ['!', '#', '%', '*', '-', '1', '2', '?', '@', 'B', 'C', 'E', 'K', 'L',
                  'Q', 'R', 'S', 'T', 'U', 'X', 'b', 'g', 'k', 'o', 'r', 't', 'y', '|']

    uni_lens = []
    for i, baseline_level_dir in enumerate(sorted(os.listdir(opt.baseline_level_dirs))):
        baseline_dataset = LevelSnippetDataset(level_dir=os.path.join(os.getcwd(), opt.baseline_level_dirs,
                                                                      baseline_level_dir,
                                                                      # "random_samples", "txt"
                                                                      ),
                                               slice_width=opt.slice_width,
                                               token_list=token_list)

        loader = DataLoader(baseline_dataset, batch_size=os.cpu_count(
        ), num_workers=os.cpu_count())
        snippets = []
        for j, (level_slices, _) in tqdm(enumerate(loader), total=len(loader)):
            level_slices = copy.deepcopy(level_slices)
            snippets.extend([level_slice.numpy().tobytes()
                             for level_slice in level_slices])

        # snippets = np.concatenate(snippets)

        uniques = np.unique(snippets)

        uni_lens.append((len(uniques), len(snippets)))
        print("Set %d - uniques %d, len %d" % (i, len(uniques), len(snippets)))

    with open(os.path.join(opt.baseline_level_dirs, "uniqueness_stats.txt"), "w") as f:
        percentages = []
        for i, (j, k) in enumerate(uni_lens):
            percentages.append(j/k)
            f.write("Set %d - uniques %d, len %d, perc %.4f\n" %
                    (i, j, k, j/k))
        f.write("Mean perc unique %.4f" % np.mean(percentages))


if __name__ == "__main__":
    main()
