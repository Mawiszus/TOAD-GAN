# TOAD-GAN

Official pytorch implementation of the paper: "TOAD-GAN: Coherent Style Level Generation from a Single Example"
For more information on TOAD-GAN, please refer to the paper (link to be added).

If you're interested in a demonstration of pre-trained generators, check out [TOAD-GUI](https://github.com/Mawiszus/TOAD-GUI).

This Project includes graphics from the game _Super Mario Bros._ **It is not affiliated with or endorsed by Nintendo.
The project was built for research purposes only.**

## Getting Started

This section includes the necessary steps to train TOAD-GAN on your system.

### Python

You will need [Python 3](https://www.python.org/downloads) and the packages specified in requirements.txt.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.
If you have a **GPU**, check if it is usable with [Pytorch](https://pytorch.org) and change the requirement in the file to use +gpu instead of +cpu.

```
$ pip3 install -r requirements.txt -f "https://download.pytorch.org/whl/torch_stable.html"
```
Make sure you use the `pip3` that belongs to your previously defined virtual environment.

## TOAD-GAN

### Training

Once all prerequisites are installed, TOAD-GAN can be started by running main.py.
Make sure you are using the python installation you installed the prerequisites into.

There are several command line options available for training. These are defined in `config.py`.
An example call which will train a 3-layer TOAD-GAN on level 1-1 of _Super Mario Bros._ would be:

```
$ python main.py --input-dir input --input-name lvl_1-1.txt --num_layer 3 --alpha 100 --niter 4000 --nfc 64 --min_nfc 64
```

### Generating samples

If you want to use your trained TOAD-GAN to generate more samples, use:
```
$ python generate_samples.py
```

### Experiments

We supply the code for experiments made for our paper.

#### TPKL-Divergence

```
$ python main_tile_pattern.py
```

#### Level Latent Space

```
$ python main_level_classification.py
```

#### Level Authoring

```
$ python generate_samples.py (with certain commands that we still need to add)
```


## Built With

* Pillow - Python Image Library for displaying images
* Pytorch - Deep Learning Framework

## Authors

* **Maren Awiszus** - Institut für Informationsverarbeitung, Leibniz University Hanover
* **Frederik Schubert** - Institut für Informationsverarbeitung, Leibniz University Hanover

## Copyright

This program is not endorsed by Nintendo and is only intended for research purposes. 
Mario is a Nintendo character which the authors don’t own any rights to. 
Nintendo is also the sole owner of all the graphical assets in the game.

## Useful Commands

These need to be reviewed an checked.

```
# Train SinGAN on every level
for file in Input/Images/SMB/*.txt; do python sin_gan/main.py --input_dir Input/Images/SMB --input_name $file:t --num_layer 3  --niter 6000 --use_scaling True --nfc 64 --min_nfc 64; done

# Compute KL-Divergence of a generator with each level individually
python sin_gan/main_tile_pattern.py --level-dir Mario-AI-Framework/levels/original --run-dir <dir_with_runs_on_level_1_till_15>

# Compute Playability for the levels in a directory
cd Mario-AI-Framework
mvn clean package
mvn exec:java -q -Dexec.mainClass=PlayLevel -Dexec.args=./levels/original
```

```
python sin_gan/main_level_classification.py --restore /home/schubert/projects/SinGAN/tmp/classifier/run-20200318_164022-n0irzrmz/checkpoints/_ckpt_epoch_8.ckpt --visualize --target-level-indices 2 7 10 --device cpu --baseline-level-dir /home/schubert/projects/SinGAN/tmp/baselines_paper
```
