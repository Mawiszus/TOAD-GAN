import os
import sys
sys.path.append(os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "..", ".."))

from typing import Tuple

import pytorch_lightning as pl
from minecraft.block2vec.block2vec import Block2Vec, Block2VecArgs
from tap import Tap


class TrainBlock2VecArgs(Block2VecArgs):
    debug: bool = False

    def process_args(self) -> None:
        super().process_args()
        os.makedirs(self.output_path, exist_ok=True)


def main():
    args = TrainBlock2VecArgs().parse_args()
    block2vec = Block2Vec(**args.as_dict())
    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, fast_dev_run=args.debug)
    trainer.fit(block2vec)


if __name__ == "__main__":
    main()
