
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from mario.level_snippet_dataset import LevelSnippetDataset

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

baseline_level_dirs = "input/umap_images/baselines"
slice_width = 5
token_list = ['!', '#', '%', '*', '-', '1', '2', '?', '@', 'B', 'C', 'E', 'K', 'L',
              'Q', 'R', 'S', 'T', 'U', 'X', 'b', 'g', 'k', 'o', 'r', 't', 'y', '|']

uni_lens = []
for i, baseline_level_dir in enumerate(sorted(os.listdir(baseline_level_dirs))):
    level_idx = None
    baseline_dataset = LevelSnippetDataset(level_dir=os.path.join(os.getcwd(), baseline_level_dirs,
                                                                  baseline_level_dir,
                                                                  # "random_samples", "txt"
                                                                  ),
                                           slice_width=slice_width,
                                           token_list=token_list, level_idx=level_idx)

    loader = DataLoader(baseline_dataset, batch_size=1, shuffle=True)
    snippets = []
    for j, s in tqdm(enumerate(loader)):
        if j >= 100000:
            break
        snippets.append(s[0].numpy().tobytes())

    uniques = np.unique(snippets)

    uni_lens.append((len(uniques), len(snippets)))
    print("Set %d - uniques %d, len %d" % (i, len(uniques), len(snippets)))

for i, (j, k) in enumerate(uni_lens):
    print("Set %d - uniques %d, len %d, perc %.4f" % (i, j, k, j/k))
