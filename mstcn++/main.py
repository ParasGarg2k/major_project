import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
# parser = argparse.ArgumentParser()
# parser.add_argument('--action', default='predict')
# parser.add_argument('--dataset', default="gtea")
# parser.add_argument('--split', default='1')
# parser.add_argument('--model_dir', default='models')
# parser.add_argument('--result_dir', default='results')

# args = parser.parse_args()

action = "predict"
query = "gtea"
split = '1'
num_epochs = 120

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
# if args.dataset == "50salads":
#     sample_rate = 2

# # To prevent over-fitting for GTEA. Early stopping & large dropout rate
# if args.dataset == "gtea":
channel_mask_rate = 0.5
    
# if args.dataset == 'breakfast':
#     lr = 0.0001


vid_list_file_tst = r"C:\Users\hp\OneDrive\Desktop\codes\major_codes\gtea\\splits\\test.split1.bundle"
vid_list_file = r"C:\Users\hp\OneDrive\Desktop\codes\major_codes\gtea\\splits\\train.split1.bundle"
features_path = "C:\\Users\\hp\\OneDrive\\Desktop\\codes\\major_codes\\gtea i3d\\"
gt_path = "C:\\Users\\hp\\OneDrive\\Desktop\\codes\\major_codes\\gtea\\ground_truth\\"
 
mapping_file = "C:\\Users\\hp\\OneDrive\\Desktop\\codes\\major_codes\\gtea\\mapping.txt"
 
model_dir = "C:\\Users\\hp\\OneDrive\\Desktop\\codes\\major_codes\\MSTCN_model\\models\\gtea\\split_1/"

results_dir = "C:\\Users\\hp\\OneDrive\\Desktop\\codes\\major_codes\\MSTCN_model\\results\\gtea2\\split_1/"
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = 11


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
if action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

