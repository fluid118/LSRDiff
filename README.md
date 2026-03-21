# LSRDiff

This repository provides the PyTorch implementation of the paper:

**"LSR-Diff: A Diffusion Model Synthesizing Level Set Representations for Reliable Segmentation of Medical Images with Ambiguous Edges."**

---

## Acknowledgements

This implementation is built upon several open-source projects. We sincerely thank the authors for their valuable contributions:

* https://github.com/openai/improved-diffusion
* https://github.com/JuliaWolleb/Diffusion-based-Segmentation
* https://github.com/suxuann/ddib

---

## Model Configuration

Following prior work, the diffusion bridge model (used to generate hybrid representations) is configured as:

```bash
--num_res_blocks 3 
--diffusion_steps 4000 
--noise_schedule linear 
--lr 1e-4
```

The diffusion-based segmentation model uses the following settings:

```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False \
--num_res_blocks 2 --num_heads 1 --learn_sigma True \
--use_scale_shift_norm False --attention_resolutions 16"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear \
--rescale_learned_sigmas False --rescale_timesteps False"

TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
```

---

## Representation Options

To compare different representations (binary masks vs. level set functions), you can use:

```bash
--target_type both
```

This allows training a **single model** that can generate both binary and LSF outputs.
Note that performance may be slightly lower, but it is sufficient for comparative analysis.

For simplified hybrid representations (binary + LSF without intermediate states):

* `--target_type label2sdf`
* `--target_type sdf2label`

(difference is discussed in the paper)

To use the **full hybrid representation** (binary + intermediate + LSF), you must:

1. Train the diffusion bridge model
2. Generate intermediate representations
3. Train the segmentation model

---

## Training & Inference

### Training

```bash
python scripts/segmentation_train.py
```

### Sampling

```bash
python scripts/segmentation_sample.py --num_ensembles 15
```

* `--num_ensembles` controls the number of repeated samplings
* Larger values generally improve performance
* Based on prior work, **15 is sufficient in most cases**

### Evaluation

```bash
python scripts/segmentation_eval.py
```

* Set `lsf=true` to enable the **EASE module**
* Set `lsf=false` to use naive thresholding

The EASE parameters can be adjusted as needed. Their theoretical meanings are explained in the paper.

---

## Data Processing

* No data augmentation is applied
* Only resizing is used
* Input resolution details are provided in the paper

---

## Notes

* This repository may contain some experimental or redundant code from earlier explorations
* We plan to further clean and reorganize the codebase

If you find redundant scripts or have suggestions for improvement, feel free to open an issue.

We may add more comments or further reorganize the codebase in the future.
If you have any suggestions or notice redundant scripts (some earlier explorations may not have been fully cleaned), please let us know. Many thanks!
