#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import yaml
import torch
import time

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from utils.config_utils import overwrite_values
from utils import register as R

########### Import your packages below ##########
from trainer import create_trainer
from data import create_dataset, create_dataloader
from utils.nn_utils import count_parameters
import models


def parse():
    parser = argparse.ArgumentParser(description='training')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # config
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configure')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    return parser.parse_known_args()


def main(args, opt_args):

    t0 = time.time()
    print_log(f'Loading config from {args.config} ...')
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    print_log(f'Config loaded in {time.time() - t0:.2f}s')

    ########## define your model #########
    print_log('Constructing model ...')
    t_model = time.time()
    model = R.construct(config['model'])
    if len(config.get('load_ckpt', '')):
        model.load_state_dict(torch.load(config['load_ckpt'], map_location='cpu').state_dict())
        print_log(f'Loaded weights from {config["load_ckpt"]}')
    print_log(f'Model ready in {time.time() - t_model:.2f}s')

    ########### load your train / valid set ###########
    print_log('Building datasets (train/valid/test) ...')
    t_data = time.time()
    train_set, valid_set, test_set = create_dataset(config['dataset'])
    print_log(
        'Datasets ready in {:.2f}s | train={} valid={} test={}'.format(
            time.time() - t_data,
            len(train_set) if train_set is not None else 0,
            len(valid_set) if valid_set is not None else 0,
            len(test_set) if test_set is not None else 0,
        )
    )

    ########## define your trainer/trainconfig #########
    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        torch.distributed.init_process_group(backend='nccl', world_size=int(os.environ['WORLD_SIZE'])) # set by torchrun
    else:
        args.local_rank = -1

    if args.local_rank <= 0:
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = create_dataloader(train_set, config['dataloader'].get('train', config['dataloader']), len(args.gpus))
    valid_loader = create_dataloader(valid_set, config['dataloader'].get('valid', config['dataloader']))
    
    trainer = create_trainer(config, model, train_loader, valid_loader)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)
    main(args, opt_args)
