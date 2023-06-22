#!/usr/bin/env python
# coding: utf-8
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
# import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.segmentation import VisTRsegm
import pycocotools.mask as mask_util
from util.box_ops import box_xyxy_to_cxcywh
##
import glob
import re
import os
from torchvision.ops import masks_to_boxes
from PIL import Image
from tqdm.notebook import tqdm
import torchvision.transforms.functional as F
from engine import train_one_epoch
import datasets.transforms as T


# In[ ]:


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=18, type=int)
    parser.add_argument('--lr_drop', default=12, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--pretrained_weights', type=str, default="r101_pretrained.pth",
                        help="Path to the pretrained model.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_queries', default=36, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--no_labels_loss', dest='labels_loss', action='store_false',
                        help="Enables labels losses")
    parser.add_argument('--no_boxes_loss', dest='boxes_loss', action='store_false',
                        help="Enables bounding box losses")
    parser.add_argument('--no_L1_loss', dest='L1_loss', action='store_false',
                        help="Enables L1 losses for bboxes")
    parser.add_argument('--no_giou_loss', dest='giou_loss', action='store_false',
                        help="Enables Generalized IOU losses for bboxes")
    parser.add_argument('--no_focal_loss', dest='focal_loss', action='store_false',
                        help="Enables Focal losses for mask")
    parser.add_argument('--no_dice_loss', dest='dice_loss', action='store_false',
                        help="Enables dice losses for mask")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='ytvos')
    parser.add_argument('--ytvos_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='r101_vistr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# In[ ]:


pat=re.compile("(\d+)\D*$")

def key_func(x):
    mat=pat.search(os.path.split(x)[-1]) # match last group of digits
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1)) # right align to 10 digits

# train_file_dir = glob.glob('./aster_updated_data_22_01_2022/Train/*')
train_file_dir = glob.glob('../Dissertation/aster_updated_data_22_01_2022/Train/*')

n_frames = 36

train_image_list = []
train_mask_list = []

for path in train_file_dir:
    frames = sorted(glob.glob(path+'/*_0001_IMAGES/images/*.jpg'), key=key_func)
    masks = sorted(glob.glob(path+'/*_0001_IMAGES/masks/*.png'), key=key_func)

#     for i in range(35):
    for i in range(len(frames)-n_frames+1):
        train_image_list.append(frames[i:i+n_frames])
        train_mask_list.append(masks[i:i+n_frames])
#         train_mask_list.append(masks[i+n_frames-1])
        
test_file_dir = glob.glob('../Dissertation/aster_updated_data_22_01_2022/Test/*')
test_image_list = []
test_mask_list = []

for path in test_file_dir:
    frames = sorted(glob.glob(path+'/*_0001_IMAGES/images/*.jpg'), key=key_func)
    masks = sorted(glob.glob(path+'/*_0001_IMAGES/masks/*.png'), key=key_func)

#     for i in range(35):
    for i in range(len(frames)-n_frames+1):
        test_image_list.append(frames[i:i+n_frames])
        test_mask_list.append(masks[i:i+n_frames])
#         test_mask_list.append(masks[i+n_frames-1])      


# In[ ]:


def make_transform(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.2316], [0.2038]) #mean #standard deviation
    ])
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([normalize])


# In[ ]:


def get_bbox(mask_list):
    return torch.cat([masks_to_boxes(mask) for mask in mask_list], dim=0)        


# In[ ]:


class ImagePathDataset(Dataset):
    def __init__(self, image_path, mask_path, n_frames, transform=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.n_frames = n_frames
        self.transform = transform
        

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        
        image = [Image.open(self.image_path[idx][i]) for i in range(self.n_frames)]
        mask = [F.to_tensor(Image.open(self.mask_path[idx][i]))
                for i in range(self.n_frames)]
        
        target = {}
        target['labels'] = torch.ones(36).long()
        target['valid'] = torch.ones(36).long()
        target['masks'] = torch.cat(mask, dim=0)
        target['boxes'] = get_bbox(mask)
        
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        image = [img.repeat(3,1,1) for img in image]
            
        return torch.cat(image,dim=0), target


# In[ ]:


def main():
    parser = argparse.ArgumentParser('VisTR training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    args.num_classes = 1
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    
    args.pretrained_weights = 'pretrained/r101.pth'
    args.masks = True
    
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # no validation ground truth for ytvos dataset
    dataset_train = ImagePathDataset(train_image_list, train_mask_list,
                                     n_frames, transform=make_transform(image_set='train'))
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)


# In[ ]:


if __name__ == '__main__':
    main()