"""
Synthetic domain translation from a source 2D domain to a target.
"""

import argparse
import os
import pathlib
import torch
import numpy as np
import torch.distributed as dist

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import model_and_diffusion_defaults
# from guided_diffusion.synthetic_datasets import scatter, heatmap, load_2d_data, Synthetic2DType
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,

    create_model_and_diffusion,

    add_dict_to_argparser,
    args_to_dict,
)
import json
from guided_diffusion.bratsmaskloader import BRATSDataset
from guided_diffusion.mask_datasets import load_mask_data
def main():
    args = create_argparser()
    logger.log(f"args: {args}")

    dist_util.setup_dist(args.device_num)
    logger.configure(dir=os.path.join('/experiments','mask'))
    logger.log("starting to sample masks.")

    code_folder = os.getcwd()
    image_folder = os.path.join(code_folder, f"experiments/images")
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    i = 'label'
    j = 'sdf'
    logger.log(f"reading models for synthetic data...")


    source_model, source_diffusion =  create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    source_model.load_state_dict(
        dist_util.load_state_dict(args.source_model_path, map_location="cpu")
    )
    source_model.to(dist_util.dev())

    target_model, target_diffusion =  create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    target_model.load_state_dict(
        dist_util.load_state_dict(args.target_model_path, map_location="cpu")
    )
    target_model.to(dist_util.dev())
    if args.use_fp16:
        source_model.convert_to_fp16()
        target_model.convert_to_fp16()

    source_model.eval()
    target_model.eval()

    sources = []
    latents = []
    targets = []
    data = load_mask_data(batch_size=args.batch_size, task='label')
    target_data = load_mask_data(batch_size=args.batch_size, task='sdf')
    ds = BRATSDataset('./training', task = 'label')

    for k, (datapice, extra) in enumerate(data):
        datapice = datapice.to(dist_util.dev())
        label = datapice[:,0,...].unsqueeze(0)
        sdf_gt = datapice[:,1,...].unsqueeze(0)
        
        logger.log(f"translating {i}->{j}, batch {k}, shape of source {label.shape}...")
        logger.log(f"device: {dist_util.dev()}")
        noise = target_diffusion.ddim_reverse_sample_loop(
            source_model, label,
            clip_denoised=False,
            device=dist_util.dev(),
        )
        gaussian_noise = torch.randn_like(label).to(dist_util.dev())
        
        logger.log(f"obtained latent representation for {label.shape[0]} samples...")
        # logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")
        
        t = torch.tensor(3999)
        x_source_t = diffusion.q_sample(source, t.to(dist_util.dev()), gaussian_noise)
        sdf_recovered = diffusion.ddim_sample_loop(
            target_model, (args.batch_size, 1, args.image_size, args.image_size),
            noise=noise,#torch.cat([label,gaussian_noise],dim = 1),#diffusion.q_sample(source,t.to(dist_util.dev()),gaussian_noise),#noise,
            clip_denoised=False,
            device=dist_util.dev(),
        )
        logger.log(f"finished translation {sdf_recovered.shape}")
        sources.append(label.cpu().numpy())
        latents.append(noise.cpu().numpy())
        targets.append(sdf_recovered.cpu().numpy())


    dist.barrier()
    logger.log(f"synthetic data translation complete: {i}->{j}\n\n")


def create_argparser():
    defaults = model_and_diffusion_defaults()
    trans_defaults = dict(
        # task=0,  # 0 to 5 inclusive
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=10,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        device_num = 1,
    )    
    defaults.update(trans_defaults)
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
