#!/usr/bin/env bash



CUDA_VISIBLE_DEVICES=4 python main.py --action=train --model=MSTCN2 --dataset=breakfast --split=1 --d_model_PG=64 --d_model_R=64 --num_layers_PG=11 --num_layers_R=10 --lr=0.0005 --weight_decay=0 --train_ratio=0.4;


CUDA_VISIBLE_DEVICES=4 python main.py --action=train --model=MSTCN2 --dataset=breakfast --split=1 --d_model_PG=64 --d_model_R=64 --num_layers_PG=11 --num_layers_R=10 --lr=0.0005 --weight_decay=0 --train_ratio=0.6;


CUDA_VISIBLE_DEVICES=4 python main.py --action=train --model=MSTCN2 --dataset=breakfast --split=1 --d_model_PG=64 --d_model_R=64 --num_layers_PG=11 --num_layers_R=10 --lr=0.0005 --weight_decay=0 --train_ratio=0.8;


CUDA_VISIBLE_DEVICES=4 python main.py --action=train --model=MSTCN2 --dataset=breakfast --split=1 --d_model_PG=64 --d_model_R=64 --num_layers_PG=11 --num_layers_R=10 --lr=0.0005 --weight_decay=0 --train_ratio=1;
