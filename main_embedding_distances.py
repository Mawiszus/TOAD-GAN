import argparse
import torch

from scipy.spatial.distance import directed_hausdorff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_1", type=str, metavar="FILE")
    parser.add_argument("embeddings_2", type=str, metavar="FILE")
    args = parser.parse_args()
    embeddings_1 = torch.load(args.embeddings_1)
    embeddings_2 = torch.load(args.embeddings_2)

    print(directed_hausdorff(embeddings_1, embeddings_2))

if __name__ == "__main__":
    main()
