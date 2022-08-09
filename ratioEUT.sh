#!/usr/bin/env bash

python main.py --action=train --dataset=breakfast --split=1 --l_seg=2000 --window_size=25 --d_model_PG=192 --d_ffn_PG=192 --n_heads_PG=6 --d_model_R=96 --d_ffn_R=96 --n_heads_R=6 --input_dropout=0.4 --ffn_dropout=0.3 --attention_dropout=0.2 --lr=0.0002 --rpe_use --rpe_share --gpu=1 --weight_decay=0.00005 --train_ratio=0.2;

python main.py --action=train --dataset=breakfast --split=1 --l_seg=2000 --window_size=25 --d_model_PG=192 --d_ffn_PG=192 --n_heads_PG=6 --d_model_R=96 --d_ffn_R=96 --n_heads_R=6 --input_dropout=0.4 --ffn_dropout=0.3 --attention_dropout=0.2 --lr=0.0002 --rpe_use --rpe_share --gpu=1 --weight_decay=0.00005 --train_ratio=0.4;

python main.py --action=train --dataset=breakfast --split=1 --l_seg=2000 --window_size=25 --d_model_PG=192 --d_ffn_PG=192 --n_heads_PG=6 --d_model_R=96 --d_ffn_R=96 --n_heads_R=6 --input_dropout=0.4 --ffn_dropout=0.3 --attention_dropout=0.2 --lr=0.0002 --rpe_use --rpe_share --gpu=1 --weight_decay=0.00005 --train_ratio=0.6;

python main.py --action=train --dataset=breakfast --split=1 --l_seg=2000 --window_size=25 --d_model_PG=192 --d_ffn_PG=192 --n_heads_PG=6 --d_model_R=96 --d_ffn_R=96 --n_heads_R=6 --input_dropout=0.4 --ffn_dropout=0.3 --attention_dropout=0.2 --lr=0.0002 --rpe_use --rpe_share --gpu=1 --weight_decay=0.00005 --train_ratio=0.8;
