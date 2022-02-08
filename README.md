# LyricsAlignment-MTL

This repository consists of code of the following paper:

Jiawen Huang, Emmanouil Benetos, Sebastian Ewert, "**Improving Lyrics Alignment through Joint Pitch Detection**," 
International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2022. [https://arxiv.org/abs/2202.01646](https://arxiv.org/abs/2202.01646)

## Dependencies

This repo is written in python 3. Pytorch is used as the deep learning framework. To install the required python packages, run

```
pip install -r requirements.txt
```

Besides, you might want to install some source-separation tool (e.g. [Spleeter](https://github.com/deezer/Spleeter), [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch))
or use your own system to prepare source-separated vocals.

## Usage

Check the [notebook](https://github.com/jhuang448/LyricsAlignment-MTL/blob/main/example.ipynb) for a quick example.

## Data

The **DALI v2.0** is required for training. See instructions on how to get the dataset: [https://github.com/gabolsgabs/DALI](https://github.com/gabolsgabs/DALI). 

To use the DALI data loader, it is recommended to pull the repo and link to the root of this repo by running:

```
ln -s path/to/dali_wrapper/ DALI
```

The annotated **Jamendo** is used for evaluation: [https://github.com/f90/jamendolyrics](https://github.com/f90/jamendolyrics) 

All the songs in both datasets need to be separated and saved in advance. 

When you run the training/testing scripts for the first time, hdf5 files will be generated.

## Training

### The baseline acoustic model (**Baseline**)

```
python train.py --dataset_dir=/path/to/DALI_v2.0/annotation/ --sepa_dir=/path/to/separated/DALI/vocals/ 
                --hdf_dir=/where/to/save/hdf5/files/
                --checkpoint_dir=/where/to/save/checkpoints/ --log_dir=/where/to/save/tensorboard/logs/ 
                --model=baseline --cuda
```

### The proposed acoustic model (**MTL**)

```
python train.py --dataset_dir=/path/to/DALI_v2.0/annotation/ --sepa_dir=/path/to/separated/DALI/mp3s/ 
                --hdf_dir=/where/to/save/hdf5/files/ --loss_w=0.5
                --checkpoint_dir=/where/to/save/checkpoints/ --log_dir=/where/to/save/tensorboard/logs/ 
                --model=MTL --cuda
```

Run `python train.py -h` for more options.

## Inference

The following script runs alignment using a pretrained baseline model without boundary information (**Baseline**) on Jamendo:

```
python eval.py --jamendo_dir=/path/to/jamendolyrics/ --sepa_dir=/path/to/separated/jamendo/mp3s/
               --load_model=./checkpoints/checkpoint_Baseline --pred_dir=/where/to/save/predictions/
               --model=baseline
```

The following script runs alignment using the pretrained MTL model with boundary information (**MTL+BDR**) on Jamendo:

```
python eval_bdr.py --jamendo_dir=/path/to/jamendolyrics/ --sepa_dir=/path/to/separated/jamendo/mp3s/
                   --load_model=./checkpoints/checkpoint_MTL --pred_dir=/where/to/save/predictions/
                   --bdr_model=./checkpoints/checkpoint_BDR --model=MTL
```

The generated csv files under `pred_dir` can be easily evaluated using the evaluation script in [jamendolyrics](https://github.com/f90/jamendolyrics).

## References

[1] Yun-Ning Hung, Yi-An Chen, and Yi-Hsuan Yang, “Multi-task learning for frame-level instrument recognition,” in Proc. ICASSP. 2019, pp. 381–385, IEEE.

[2] Sebastian Ewert, Meinard Müller, and Peter Grosche, “High resolution audio synchronization using chroma onset features,” in Proc. ICASSP. 2009, pp. 1869–1872, IEEE.

[3] Daniel Stoller, Simon Durand, and Sebastian Ewert, “End-to-end lyrics alignment for polyphonic music using an audio-to-character recognition model,” in Proc. ICASSP. 2019, pp. 181–185, IEEE.

[4] Gabriel Meseguer-Brocal, Alice Cohen-Hadria, and Geoffroy Peeters, “Creating DALI, a large dataset of synchronized audio, lyrics, and notes,” Transactions of the International Society for Music Information Retrieval, vol. 3, no. 1, pp. 55–67, 2020.

[5] Chitralekha Gupta, Emre Yılmaz, and Haizhou Li, “Automatic lyrics alignment and transcription in polyphonic music: Does background music help?,” in Proc. ICASSP. 2020, pp. 496–500, IEEE.


## Contact

Jiawen Huang

jiawen.huang@qmul.ac.uk
