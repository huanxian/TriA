# TriA

## Environment Preparation

First, install the dependency libraries:

```shell
conda env create -f environment.yml
```

Then download the required model checkpoint and place it in the `ckpt/`:

```shell
## BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt
wget https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC
# or download from https://github.com/microsoft/unilm/tree/master/beats

## audiobox-aesthetics/checkpoint.pt
huggingface-cli download facebook/audiobox-aesthetics

## CLAP_weights_2023.pth
huggingface-cli download microsoft/msclap
```

## Quick Start

```shell
python3 main.py --input_folder_path /path_to_raw_audio_directory
```

The default filtering thresholds for PC, PQ and CLAP Similarity are 2.24, 5.85 and 7.43 respectively, which are obtained by statistically analyzing datasets related to the domestic audio classification task. You can set a more appropriate filtering threshold according to the target task:

```shell
python3 main.py --input_folder_path /path_to_raw_audio_directory --pc_threshold 2.24 --pq_threshold 5.85 --clap_threshold 7.43
```

## TODO

- [x] Release the source code of the TriA Pipeline.
- [ ] Release Pipeline parameter configuration guide
- [ ] Release the TriA dataset.

## Acknowledgement

We borrow a lot of code from [Emilia](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia), [BEATs](https://github.com/microsoft/unilm/tree/master/beats)...