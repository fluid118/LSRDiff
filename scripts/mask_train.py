"""
Train a diffusion model on 2D synthetic datasets.
"""
import sys
import os
sys.path.append('..')
sys.path.append('.')

import argparse
import torch
import json
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import model_and_diffusion_defaults_2d, create_model_and_diffusion_2d, args_to_dict, \
    add_dict_to_argparser,create_model_and_diffusion, model_and_diffusion_defaults
from guided_diffusion.synthetic_datasets import Synthetic2DType, load_2d_data
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.mask_datasets import load_mask_data
from guided_diffusion.bratsmaskloader import BRATSDataset
def main():
    args = create_argparser()

    dist_util.setup_dist(args.device_num)

    logger.configure(dir=os.path.join('./models/mask/condition',args.task))
    logger.log(f"args: {vars(args)}")  # 适用于 `args` 是 `dict` 或 `list`
    logger.log("creating MASK model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating MASK data loader...")

    data = load_mask_data(batch_size=args.batch_size, task=args.task)
    ds = BRATSDataset('/training', task = args.task)

    datal= torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,)
    data = iter(datal)
    logger.log("training MASK model...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        dataloader = datal,
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
    ).run_loop()


def create_argparser():
    defaults = model_and_diffusion_defaults()
    
    train_defaults = dict(
        task = 'label',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=10,
        microbatch=-1,  
        ema_rate="0.9999",  
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        device_num = 1,
    )

    defaults.update(train_defaults)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON configuration file")
    args, remaining_argv = parser.parse_known_args()  


    if args.config:
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                config_params = json.load(f)
                defaults.update(config_params) 
        else:
            print(f"Warning: Config file {args.config} not found. Using default parameters.")


    parser = argparse.ArgumentParser()
    for key, val in defaults.items():
        arg_type = type(val) if val is not None else str  
        parser.add_argument(f"--{key}", type=arg_type, default=val)

    args = parser.parse_args(remaining_argv)  
    return args


if __name__ == "__main__":
    main()
