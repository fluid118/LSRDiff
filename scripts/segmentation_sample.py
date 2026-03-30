"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import torchvision
import my_transforms as trs
from torchvision import transforms
import argparse
import os
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=8850) #8850
import sys
import random
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion_backup import dist_util, logger
from guided_diffusion_backup.bratsloader import BRATSDataset
from guided_diffusion_backup.ISIC18loader import ISIC18Dataset
from guided_diffusion_backup.REFUGE2loader import REFUGE2Dataset, REFUGE2DatasetCup
from guided_diffusion_backup.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.device_num)
    logger.configure(dir = "./archive")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.dataset == 'BRATS':
        ds = BRATSDataset(args.data_dir, test_flag=True)
    elif args.dataset == 'ISIC':
        data_transform = torchvision.transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            ])
        ds = ISIC18Dataset(args.data_dir, test_flag=True, transform = data_transform) #修改
    elif args.dataset == 'REFUGE2Cup':
        data_transform = torchvision.transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),            
            ]
    )
        ds = REFUGE2DatasetCup(args.data_dir, test_flag=True, transform=data_transform)
    elif args.dataset == "REFUGE2":
        data_transform = torchvision.transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),            
            ]
    )
        ds = REFUGE2Dataset(args.data_dir, test_flag=True, transform=data_transform)

    
    else:
        AssertionError('Not Implemented Dataset')
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    folder_path = os.path.join(os.path.dirname(args.model_path), 'sampling_'+os.path.basename(args.model_path)[-9:-7]+'w')
    os.makedirs(folder_path, exist_ok=True)

    while len(all_images) * args.batch_size < args.num_samples:
        b, path = next(data)  
        if args.target_type in {'label2sdf', 'label', 'sdf', 'cliff', 'sdf2label'}:
            target_number = 1
        elif args.target_type in {'both', 'label2sdf_refuge', 'label_refuge'}: # multi_class label
            target_number = 2

        noise_template = th.zeros((b.shape[0], target_number, b.shape[2], b.shape[3]))
        c = th.randn_like(noise_template)

        img = th.cat((b, c), dim=1)      
        if args.dataset == 'BRATS':
            slice_ID=path[0].split("/")[-1].split("_")[2]+"_"+path[0].split("/")[-1].split("_")[4]
        elif args.dataset == 'ISIC':
            slice_ID = path[0].split("/")[-1].split("_")[1].split('.')[0]
        elif args.dataset in {'REFUGE2', 'REFUGE2Disc', 'REFUGE2Cup'} :
            slice_ID = path[0].split("/")[-1].split("-")[0]   
        else:
            AssertionError('Not Implemented Dataset')

        file_path = os.path.join(folder_path, slice_ID+'_output')
        if os.path.exists(file_path):
            continue

        for i in range(img.shape[1]):
            viz.image(visualize(img[0, i,...]), opts=dict(caption="img input"+str(i)))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        res = {}
        ensembles = [] 
        print(file_path)
        print('---------------')
        print(args.target_type)
        print(args.dataset)
        print("device: ", args.device_num)
        print('---------------')
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            conc_sample = th.cat((sample["sample"], sample["mean"], sample["log_var"]), dim=1)
            ensembles.append(conc_sample) 

        res = th.cat(ensembles, dim = 0) 

        th.save(res, file_path) #save the generated mask 

def create_argparser():
    defaults = dict(
        dataset = "BRATS",
        data_dir="",
        clip_denoised=False,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=15, 

        condition_channel = 4,
        device_num = 3,
        target_type = "label2sdf",
        scale_sdf = 1,
    )
    
    parser = argparse.ArgumentParser()
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)

    
    args = parser.parse_args()
    target_type = args.model_path.split('/')[6].split('-')[0]
    dataset = args.model_path.split('/')[5]
    scale_sdf = args.model_path.split('/')[6].split('-')[1][1]

    parser.set_defaults(target_type=target_type)
    parser.set_defaults(dataset=dataset)
    parser.set_defaults(scale_sdf=scale_sdf)
    
    return parser


if __name__ == "__main__":

    main()
