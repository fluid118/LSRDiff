import copy
import functools
import os
import time

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler,LossSecondMomentResampler
try:
    from visdom import Visdom
    viz = Visdom(port=8850)
except Exception:
    viz = None

from scripts.level_set import gradient_sobel

# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        condition_channel,
        target_type,
    ):
        self.model = model
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.c = condition_channel
        self.t = target_type

        # Optional training benchmark (timing + peak memory). Enable via env vars so
        # you can run training normally and just stop manually after collecting enough logs.
        self.bench_enabled = bool(os.environ.get("DIFFUSION_TRAIN_BENCH", ""))
        self.bench_warmup_iters = int(os.environ.get("DIFFUSION_TRAIN_BENCH_WARMUP_ITERS", "5"))
        self.bench_log_every = int(os.environ.get("DIFFUSION_TRAIN_BENCH_LOG_EVERY_ITERS", "20"))
        self.bench_seconds = float(os.environ.get("DIFFUSION_TRAIN_BENCH_SECONDS", "0"))  # 0 => no auto-stop
        self.perf_out_dir = os.environ.get("DIFFUSION_TRAIN_BENCH_OUT_DIR", "/work/gaowenbo/Diffusion-Cross/perf_logs")
        self._bench_started = False
        self._bench_iter_times = []
        self._bench_peak_mem_bytes = 0
        self._bench_wall_start = None
        self._bench_last_log_time = None
        self._bench_rank = None
        try:
            self._bench_rank = dist.get_rank()
        except Exception:
            self._bench_rank = 0

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        if self.bench_enabled and th.cuda.is_available() and self._bench_rank == 0:
            os.makedirs(self.perf_out_dir, exist_ok=True)
            bench_tag = f"train_bench_step{self.resume_step:06d}"
            self._bench_out_csv = os.path.join(self.perf_out_dir, f"{bench_tag}.csv")
            if not os.path.exists(self._bench_out_csv):
                with open(self._bench_out_csv, "w") as f:
                    f.write("step,iter_time_s,iter_time_s_mean,peak_mem_gb,wall_time_s\n")

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    # batch, cond = next(data_iter)
                    sample = next(data_iter)
                    batch = sample['img']
                    label = sample['gt']
                    df = sample['sdfs'] #distance function
                    vx, vy = gradient_sobel(df) #vector field
                    sdf = sample['trunc_sdfs']#用sdf替代gt 修改
                    cliff = sample['cliff_sdfs']#用sdf替代gt 修改


                    if self.t == 'label':
                        # cond = th.cat((label, sdf), dim=1)
                        cond = label
                    elif self.t == 'sdf':
                        cond = sdf
                    elif self.t == 'label2sdf':
                        cond = sdf

                    elif self.t == 'label2sdf_ACDC':
                        sdf1 = sample['trunc_sdfs1']
                        sdf2 = sample['trunc_sdfs2']
                        sdf3 = sample['trunc_sdfs3']
                        cond = th.cat((sdf1, sdf2, sdf3), dim=1)
                    elif self.t == 'label2sdf_refuge':
                        sdf1 = sample['trunc_sdfs1']
                        sdf2 = sample['trunc_sdfs2']
                        # sdf3 = sample['trunc_sdfs3']
                        cond = th.cat((sdf1, sdf2), dim=1)
                    elif self.t == 'label_refuge':
                        label1 = sample['mask1']
                        label2 = sample['mask2']
                        cond = th.cat((label1, label2), dim=1)
                    elif self.t == 'label_ACDC':
                        label1 = sample['mask1']
                        label2 = sample['mask2']
                        label3 = sample['mask3']
                        cond = th.cat((label1, label2, label3), dim=1)
                    elif self.t == 'sdf2label':
                        cond = sdf
                    elif self.t == 'cliff':
                        cond = cliff
                    elif self.t == 'both':
                        cond = th.cat((label, sdf), dim=1)
                    else:
                        raise NotImplementedError(f"unknown beta schedule: {self.t}")
                    # cond = th.cat((label, sdf), dim=1) #修改
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader)
                    # batch, cond = next(data_iter)
                    sample = next(data_iter)
                    batch = sample['img']
                    label = sample['gt']
                    df = sample['sdfs'] #distance function
                    vx, vy = gradient_sobel(df) #vector field
                    sdf = sample['trunc_sdfs'] #修改
                    if self.t == 'label':
                        cond = label
                    elif self.t == 'sdf':
                        cond = sdf
                    elif self.t == 'label2sdf':
                        cond = sdf
                        
                    elif self.t == 'label2sdf_ACDC':
                        sdf1 = sample['trunc_sdfs1']
                        sdf2 = sample['trunc_sdfs2']
                        sdf3 = sample['trunc_sdfs3']
                        cond = th.cat((sdf1, sdf2, sdf3), dim=1)
                    elif self.t == 'label2sdf_refuge':
                        sdf1 = sample['trunc_sdfs1']
                        sdf2 = sample['trunc_sdfs2']
                        # sdf3 = sample['trunc_sdfs3']
                        cond = th.cat((sdf1, sdf2), dim=1)
                    elif self.t == 'label_refuge':
                        label1 = sample['mask1']
                        label2 = sample['mask2']
                        cond = th.cat((label1, label2), dim=1)
                    elif self.t == 'label_ACDC':
                        label1 = sample['mask1']
                        label2 = sample['mask2']
                        label3 = sample['mask3']
                        cond = th.cat((label1, label2, label3), dim=1)
                    elif self.t == 'sdf2label':
                        cond = sdf
                    elif self.t == 'both':
                        cond = th.cat((label, sdf), dim=1)
                    else:
                        raise NotImplementedError(f"unknown beta schedule: {self.t}")
                    # cond = th.cat((label, sdf), dim=1) #修改
            # Benchmark: measure iter wall time around the core compute (run_step).
            bench_t0 = time.perf_counter() if self.bench_enabled else None
            self.run_step(batch, cond)
            bench_t1 = time.perf_counter() if self.bench_enabled else None

            if self.bench_enabled:
                curr_step = self.step + self.resume_step

                if (not self._bench_started) and curr_step >= self.bench_warmup_iters:
                    self._bench_started = True
                    if th.cuda.is_available():
                        th.cuda.reset_peak_memory_stats()
                    self._bench_iter_times = []
                    self._bench_peak_mem_bytes = 0
                    self._bench_wall_start = time.perf_counter()
                    self._bench_last_log_time = self._bench_wall_start

                if self._bench_started:
                    # iter time
                    iter_time_s = bench_t1 - bench_t0
                    self._bench_iter_times.append(iter_time_s)

                    # peak mem across ranks (MAX)
                    peak_bytes_local = 0
                    if th.cuda.is_available():
                        peak_bytes_local = th.cuda.max_memory_allocated()
                    peak_tensor = th.tensor([peak_bytes_local], device=dist_util.dev())
                    try:
                        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
                    except Exception:
                        pass
                    peak_bytes = int(peak_tensor.item())
                    self._bench_peak_mem_bytes = max(self._bench_peak_mem_bytes, peak_bytes)

                    # logging
                    if (len(self._bench_iter_times) % self.bench_log_every) == 0 and self._bench_rank == 0:
                        iter_mean = float(sum(self._bench_iter_times) / len(self._bench_iter_times))
                        peak_gb = self._bench_peak_mem_bytes / (1024 ** 3)
                        wall_s = time.perf_counter() - self._bench_wall_start
                        with open(self._bench_out_csv, "a") as f:
                            f.write(f"{curr_step},{iter_time_s},{iter_mean},{peak_gb},{wall_s}\n")

                    # optional auto-stop
                    if self.bench_seconds > 0 and (time.perf_counter() - self._bench_wall_start) >= self.bench_seconds:
                        return
          
            i += 1          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        batch=th.cat((batch, cond), dim=1)

        cond={}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            micro[0,-1,:,:].cpu().numpy()
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev()) #生成随机t和对应的权重w_t

            compute_losses = functools.partial( #计算损失函数
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            ) #return (terms, model_output)

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            losses = losses1[0]
            sample = losses1[1]

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(#
                    t, losses["loss"].detach()
                )


            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            return  sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
