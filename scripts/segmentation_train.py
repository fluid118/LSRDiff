"""
Train a diffusion model on images.
"""

import sys
import os
sys.path.append("..")
sys.path.append(".")

import argparse
import torchvision
import my_transforms as trs

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.ISIC18loader import ISIC18Dataset
from guided_diffusion.LiTSloader import LiTSDataset
from guided_diffusion.REFUGE2loader import  REFUGE2DatasetCup, REFUGE2DatasetDisc


from thop import profile


from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.device_num)
    logger.configure(dir = os.path.join("./results", args.dataset, args.target_type+'-Cross'))

    logger.log("creating model and diffusion...")

    

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)


    # input_tensor = th.randn(10, 4, 224, 224)
    # flops, params = profile(model, inputs=(input_tensor, th.tensor(10))) 
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs") 
    # print(f"Params: {params / 1e6:.2f} M")

    logger.log("creating data loader...")
    if args.dataset == 'BRATS':
        data_transform = torchvision.transforms.Compose(
            [
                trs.SDF(scale=args.scale_sdf)
             ])
        ds = BRATSDataset(args.data_dir, test_flag=False, transform = data_transform) 

    elif args.dataset == 'ISIC':
        data_transform = torchvision.transforms.Compose(
            [
                trs.CustomResize((256, 256)),
                trs.CustomToTensor(),
                trs.SDF(scale=args.scale_sdf),
            ])
        ds = ISIC18Dataset(args.data_dir, test_flag=False, transform = data_transform) 

    elif args.dataset == "LiTS":
        data_transform = torchvision.transforms.Compose(
            [
                trs.SDF(scale=args.scale_sdf),
            ])
        ds = LiTSDataset(data_dir=args.data_dir, test_flag=False, transform=data_transform)
    elif args.dataset == 'REFUGE2Cup':
        data_transform = torchvision.transforms.Compose(
            [
                trs.CustomResize((256, 256)),
                trs.CustomToTensor(),
                trs.SDF(scale=args.scale_sdf),                
            ]
    )
        ds = REFUGE2DatasetCup(args.data_dir, test_flag=False, transform=data_transform)

        ds = REFUGE2DatasetDisc(args.data_dir, test_flag=False, transform=data_transform)        
    else:
        AssertionError('Not Implemented Dataset')

    logger.log("size of dataset: "+str(len(ds)))
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,)
    data = iter(datal)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        condition_channel = args.condition_channel,
        target_type = args.target_type,
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset = "BRATS", # avaliable: brats, isic, refuge_cup
        data_dir=".../training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=400000, #修改 迭代次数0
        batch_size=10,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,#100,
        save_interval=10000,
        resume_checkpoint='', # load ckpt
        use_fp16=False,
        fp16_scale_growth=1e-3,
        condition_channel = 4, # concanate channel of lable (BRATS+mask)
        device_num = 0,
        target_type = "label2sdf",
        scale_sdf = 1,
        cross_loss = False,

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
