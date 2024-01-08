# Do we really need temporal convolutions in action segmentation?

This repository is the official implementation of 2023 ICME paper: **[Do we really need temporal convolutions in action segmentation?](https://arxiv.org/abs/2205.13425)**.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

Since the dataset files are too large, please download them yourself on a public website. To prepare the datasets as follows:

- Download the three public datasets `data.zip` at (https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) or (https://zenodo.org/record/3625992#.Xiv9jGhKhPY), including the video feature and frame-wise labels. These datasets have been already preprocessed and splited.
- Unzip the `data.zip` to the current folder. There are three datasets in the `./data` folder, i.e. `./data/breakfast`, `./data/50salads`, `./data/gtea`.

## Training

To train the model(s) in the paper, run this command:

```train
# breakfast and gtea have 4 splits and 50salads has 5 splits
python main.py --action=train --dataset=50salads/gtea/breakfast --split=1/2/3/4/5 
```

To achieve results of the 50Salads dataset, run this command:

```train
python main.py --action=train --dataset=50salads --split=1/2/3/4/5 --pg_layers=5 --r_layers=5 --num_R=3 --l_seg=200 --window_size=51 --d_model_PG=128 --d_ffn_PG=128 --n_heads_PG=4 --d_model_R=64 --d_ffn_R=64 --n_heads_R=4 --bz=1 --activation=relu --input_dropout=0.4 --ffn_dropout=0.3 --attention_dropout=0.2 --num_epochs=150
```

## Predicting

After training the model(s), run this command to predict the results of the test split:

```predict
python main.py --action=predict --dataset=50salads/gtea/breakfast --split=1/2/3/4/5 --num_epochs=100
```

where `num_epochs` refers to the training model at which epoch to use for prediction.

To predict results of the 50Salads dataset, run this command:

```predict
python main.py --action=predict --dataset=50salads --split=1/2/3/4/5 --pg_layers=5 --r_layers=5 --num_R=3 --l_seg=200 --window_size=51 --d_model_PG=128 --d_ffn_PG=128 --n_heads_PG=4 --d_model_R=64 --d_ffn_R=64 --n_heads_R=4 --bz=1 --activation=relu --input_dropout=0.4 --ffn_dropout=0.3 --attention_dropout=0.2 --num_epochs=150
```

## Evaluation

To evaluate the prediction of the model on the dataset, run:

```eval
python eval.py --dataset=50salads/gtea/breakfast --split=0
```

The above command is to evaluate the average performance over all splits. You can also eval the model at any split:

```eval
python eval.py --dataset=50salads/gtea/breakfast --split=1/2/3/4/5
```

## Pre-trained Models

We provide pretrained models in the `./all_results_and_model/dataset/checkpoints` folder. 

You can directly run this command to predict results of the 50Salads dataset using the pretrained models (without training from scratch):

```predict
python main.py --action=predict --dataset=50salads --split=1/2/3/4/5 --pg_layers=5 --r_layers=5 --num_R=3 --l_seg=200 --window_size=51 --d_model_PG=128 --d_ffn_PG=128 --n_heads_PG=4 --d_model_R=64 --d_ffn_R=64 --n_heads_R=4 --bz=1 --activation=relu --input_dropout=0.4 --ffn_dropout=0.3 --attention_dropout=0.2 --num_epochs=150
```

## Citation
```
@inproceedings{du2023we,
  title={Do We Really Need Temporal Convolutions in Action Segmentation?},
  author={Du, Dazhao and Su, Bing and Li, Yu and Qi, Zhongang and Si, Lingyu and Shan, Ying},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1014--1019},
  year={2023},
  organization={IEEE}
}
```
