import torch
from exp.exp_EUT import ExpEUT
from exp.exp_other import ExpOther
import os
import argparse
import random

seed = 19990605
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='EUT')
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--root_path', default='./data/')

# hyper-parameter
parser.add_argument('--window_size', type=int, default=31, help='window size')
parser.add_argument('--l_seg', type=int, default=200, help='the length of segment')

parser.add_argument('--input_dim', type=int, default=2048, help='dim of input')

parser.add_argument('--d_model_PG', type=int, default=64, help='hidden dimension')
parser.add_argument('--d_ffn_PG', type=int, default=128, help='ffn dimension')
parser.add_argument('--n_heads_PG', type=int, default=4, help='heads num')

parser.add_argument('--d_model_R', type=int, default=32, help='hidden dimension')
parser.add_argument('--d_ffn_R', type=int, default=64, help='ffn dimension')
parser.add_argument('--n_heads_R', type=int, default=4, help='heads num')

parser.add_argument('--input_dropout', type=float, default=0.3, help='dropout2d rate')
parser.add_argument('--attention_dropout', type=float, default=0.2, help='dropout rate in attention')
parser.add_argument('--ffn_dropout', type=float, default=0.2, help='dropout rate in ffn')

parser.add_argument('--pre_norm', action="store_true", help='pre or post layernorm, default False')
parser.add_argument('--rpe_share', action="store_true", help='PG and R stage share rpe, default False')
parser.add_argument('--rpe_use', action="store_true", help='use rpe or not, default False')
parser.add_argument('--activation', type=str, default='relu', help='activation')

parser.add_argument('--num_layers_PG', type=int, default=5, help='prediction generation layer num')
parser.add_argument('--num_layers_R', type=int, default=5, help='refinement layer num')
parser.add_argument('--num_R', type=int, default=3, help='refinement stage num')

parser.add_argument('--gamma', type=float, default=0.15, help='tmse loss weight')
parser.add_argument('--beta', type=float, default=0.15, help='boundary-aware loss weight')
parser.add_argument('--baloss', action="store_true", help='use boundary-aware loss or not, default False')

# training-parameter
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--adambeta', type=float, default=0.98)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--bz', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

args.model_dir = model_dir = "./checkpoints/"+args.dataset+"/split_"+args.split
args.results_dir = results_dir = "./results/"+args.dataset+"/split_"+args.split
args.attn_dir = attn_dir = "./attn/"+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(attn_dir):
    os.makedirs(attn_dir)

dataset2numclasses = {"gtea": 11, "50salads": 19, "breakfast": 48}
args.num_classes = dataset2numclasses[args.dataset]

if args.model == 'EUT':
    my_exp = ExpEUT(configs=args)
else:
    my_exp = ExpOther(configs=args)

if args.action == "train":
    my_exp.train()

if args.action == "predict":
    my_exp.predict()

# if args.action == "length":
#     batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
#     batch_gen.read_data(vid_list_file)
#     print("train samples")
#     train_num = 0
#     train_len = 0
#     while batch_gen.has_next():
#         batch_input_tensor, _, _, _, batch_length, _, batch_target_source = batch_gen.next_batch(1)
#         train_num += 1
#         print(batch_length[0])
#         train_len += batch_length[0]
#     print("train num: {}".format(train_num))
#     print("average train length: {}".format(train_len/train_num))

#     test_num = 0
#     test_len = 0
#     batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
#     batch_gen_tst.read_data(vid_list_file_tst)
#     print("test samples")
#     while batch_gen_tst.has_next():
#         batch_input_tensor, _, _, _, batch_length, _, batch_target_source = batch_gen_tst.next_batch(1)
#         test_num += 1
#         print(batch_length[0])
#         test_len += batch_length[0]
#     print("test num: {}".format(test_num))
#     print("average test length: {}".format(test_len / test_num))

# if args.action == "friedegg":
#     batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
#     batch_gen.read_data(vid_list_file_tst)
#     lis = []
#     lis1 = []
#     while batch_gen.has_next():
#         batch_input, _, mask, vids, batch_length, batch_chunk, batch_target_source = batch_gen.next_batch(1)
#         if 'P16_cam01_P16_friedegg' in vids[0]:
#             print(vids[0])
#             for i in range(0, len(batch_target_source[0]), 5):
#                 lis1.append(reverse_dict[batch_target_source[0][i]])
#             lis.append(lis1)
#             break
#     print(lis)
