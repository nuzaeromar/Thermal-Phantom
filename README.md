# ThermalMirage

ThermalMirage generates adversarial thermal images for FLIR object detection using class conditional GAN priors, trained object diffusion, analytic context edits, and detector guided refinement.

The main experiment entry point is `run_multiclass.py`; `run.py` provides GAN training and the older single-strategy workflow.

## 1. Set up the environment

From the workspace root:

```bash
cd thermal_mirage
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install tqdm
mkdir -p checkpoints results logs
export FLIR_ROOT="/absolute/path/to/FLIR_ADAS_1_3"
```

`tqdm` is imported by the current runners but is not explicitly listed in `requirements.txt`. For optional LPIPS measurements, install `lpips` and add `--compute_lpips` to the attack or ablation command:

```bash
python -m pip install lpips
```

Use a CUDA-enabled PyTorch installation for GPU runs. Check the environment before starting training:

```bash
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

The examples use `--device cuda`. Use `--device cpu` when CUDA is unavailable; the multi-class runner accepts only `cuda` or `cpu`. Training and full-image attacks can be expensive on CPU. `train_diffusion.py` uses `torch.amp.GradScaler("cuda", ...)`, so the installed PyTorch must support that API; the loose minimum in `requirements.txt` alone does not establish compatibility. Initial pretrained detector weights may require a download.

## 2. Prepare the dataset

Set `FLIR_ROOT` to the dataset root, containing both splits:

```text
FLIR_ADAS_1_3/
├── train/
│   ├── thermal_8_bit/
│   │   └── ... thermal images ...
│   └── thermal_annotations
└── val/
    ├── thermal_8_bit/
    │   └── ... thermal images ...
    └── thermal_annotations
```

Annotations are COCO-style JSON, even when their filenames have no extension. The attack loader also probes `thermal annotations.json`, `thermal_annotations.json`, `annotations/instances_{split}.json`, and root-level `images_{split}.json`. For the alternate `annotations/` layout, images can be placed in `images/train/` and `images/val/`. Detector-training scripts have their own annotation probes; the layout above works across the documented workflow.

Use the thermal images and their annotations. The default full-image size is **640 × 512 pixels**, represented as `(H, W) = (512, 640)` in the code.

| Internal class ID | Class   | FLIR ADAS 1.3 category ID |
| ----------------- | ------- | ------------------------- |
| 0                 | person  | 1                         |
| 1                 | bicycle | 2                         |
| 2                 | car     | 3                         |
| 3                 | dog     | 17                        |

The detector wrapper handles the Faster R-CNN background-label offset internally. Pass internal IDs to single-strategy attack arguments and class names to `--attack_classes`.

## 3. Train or supply the detector checkpoints

Skip training for any compatible FLIR checkpoint you already have. Detectors must use the four classes in the order above. Generic COCO weights alone do not provide a trained FLIR classification head.

### Faster R-CNN

Edit `data_root` in `fasterR-CNN.py` to your dataset path. This script has no command-line parser; passing `--data_root` will not configure it.

```bash
python fasterR-CNN.py
```

It trains for 20 epochs and saves `checkpoints/fasterrcnn_flir.pt`. The multi-class setup constructs Faster R-CNN even when evaluating YOLO or RT-DETR, because its feature extractor builds the GAN class priors. Supply `--detector_ckpt` in those runs too.

### YOLOv11, for YOLO evaluation

In this workspace, the YOLO trainer lives outside `thermal_mirage/`:

```bash
python ../Results/yolov11_fastrcnn/train_yolov11.py \
  --data_root "$FLIR_ROOT" \
  --model yolo11s.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device cuda \
  --workers 4 \
  --checkpoint_dir checkpoints
```

The script prepares YOLO labels and copies the best weights to `checkpoints/yolo11_flir.pt`. Keep the sibling `Results/` directory available when using this command.

### RT-DETR, for RT-DETR evaluation

```bash
python train_rtdetr_flir.py \
  --data_root "$FLIR_ROOT" \
  --model rtdetr-l.pt \
  --epochs 50 \
  --batch 8 \
  --device cuda \
  --checkpoint_dir checkpoints
```

This saves the best checkpoint as `checkpoints/rtdetr_flir.pt`.

## 4. Train the class-conditional thermal GAN

```bash
python run.py \
  --mode train_gan \
  --data_root "$FLIR_ROOT" \
  --epochs 100 \
  --device cuda \
  --out_dir checkpoints
```

The current training function extracts **64 × 64** object crops and uses batch size 32. It saves generator and discriminator weights to `checkpoints/thermal_gan.pt`. The multi-class runner requires this checkpoint, including when diffusion is omitted.

Inspect generated samples:

```bash
python run_test.py
```

This script uses CUDA and the fixed checkpoint path above. Change its `device` variable for CPU. It saves **`checkpoints/gan_priors_check.png`**, with rows for person, bicycle, car, and dog. Check for varied, class-dependent thermal structure. Training time and loss values depend on the data and hardware; loss magnitude alone does not demonstrate convergence.

## 5. Train object diffusion

Train this model to enable the learned object diffusion branch:

```bash
python train_diffusion.py \
  --data_root "$FLIR_ROOT" \
  --out_dir checkpoints/object_diffusion \
  --image_size 128 \
  --epochs 100 \
  --batch_size 16 \
  --num_workers 4 \
  --base_ch 64 \
  --channel_mult 1,2,4,8 \
  --num_res_blocks 2 \
  --attn_resolutions 16 \
  --cond_dim 64 \
  --timesteps 1000 \
  --device cuda
```

Outputs include `best.pt`, `latest.pt`, periodic `epoch_*.pt` checkpoints, and `training_config.json`. The complete checkpoints contain the U-Net, learned class embeddings, architecture settings, and scheduler settings needed by the attack runner.

To resume, rerun the same training command with:

```text
--resume checkpoints/object_diffusion/latest.pt
```

Keep architecture and scheduler arguments consistent with the original run. `--epochs` is the total training target, including completed epochs.

To use the gradient fallback, skip this training step and omit `--diffusion_ckpt` from multi-class commands. The full engine then uses its FGSM object fallback where needed and analytic context edits. Supplying a nonexistent or incompatible diffusion checkpoint causes initialization to fail. Confirm the log reports `Full attack engine ready`; a `SimpleAttackWrapper` fallback indicates a different, simplified execution path.

## 6. Run a small adaptive attack

Start with a short validation run against Faster R-CNN:

```bash
python run_multiclass.py \
  --mode attack_adaptive \
  --data_root "$FLIR_ROOT" \
  --gan_ckpt checkpoints/thermal_gan.pt \
  --diffusion_ckpt checkpoints/object_diffusion/best.pt \
  --detector_ckpt checkpoints/fasterrcnn_flir.pt \
  --loop_detector fasterrcnn \
  --target_detector fasterrcnn \
  --split val \
  --num_images 10 \
  --batch_size 1 \
  --device cuda \
  --context_attack sequential \
  --context_strength 0.12 \
  --max_perturbation 0.20 \
  --n_refinement_steps 8 \
  --save_images \
  --verbose \
  --out_dir results/adaptive_fasterrcnn
```

Adaptive mode tests strategies per object and composes the selected edits into the image. Remove `--num_images 10` to process the full selected split. Add `--attack_classes person` or, for example, `--attack_classes person car` to restrict classes.

The example explicitly sets a final L∞ budget of `0.20` on image values in `[0, 1]`; the runner default is `0.15`. The default context strength is `0.08`, and the default number of refinement steps is `3`. Record the chosen settings when comparing experiments.

### Choose guidance and evaluation detectors

`--loop_detector` selects the detector used for gradient guidance. `--target_detector` selects the detector used for final success and detection metrics.

To evaluate the command above on YOLOv11, replace its target option with:

```text
--target_detector yolo --yolo_ckpt checkpoints/yolo11_flir.pt
```

For RT-DETR evaluation, use:

```text
--target_detector rtdetr --rtdetr_ckpt checkpoints/rtdetr_flir.pt
```

To also guide optimization with YOLO, replace the loop option with:

```text
--loop_detector yolo --loop_yolo_ckpt checkpoints/yolo11_flir.pt
```

RT-DETR is supported as an evaluation target, but is not a loop-detector option. Strategy selection also scores candidates with the configured evaluation detector, so adaptive selection can use feedback from that detector even when gradients come from another model.

The default `sequential` context operation applies contrast reduction, thermal smoothing, and adaptive cooling. Other `--context_attack` choices are `contrast`, `smooth`, `cool`, `adaptive_cool`, `night`, `low_contrast`, `blur`, and `clutter`. Use this flag to select context editing; `--scene_cond` is retained for backward compatibility.

## 7. Profile strategies and run profiled mode

Profiling is optional. Adaptive mode does not require a profile. Generate a profile using the same checkpoints, detector choices, and attack settings intended for the later run:

```bash
python run_multiclass.py \
  --mode profile \
  --data_root "$FLIR_ROOT" \
  --gan_ckpt checkpoints/thermal_gan.pt \
  --diffusion_ckpt checkpoints/object_diffusion/best.pt \
  --detector_ckpt checkpoints/fasterrcnn_flir.pt \
  --profile_samples 150 \
  --profile_path checkpoints/strategy_profile.json \
  --context_attack sequential \
  --context_strength 0.12 \
  --max_perturbation 0.20 \
  --n_refinement_steps 8 \
  --device cuda
```

Profiling always uses the validation split. The profile is saved to `--profile_path`, rather than being derived from `--out_dir`. For reporting, account for the overlap between profiling and evaluation data.

```bash
python run_multiclass.py \
  --mode attack_profiled \
  --data_root "$FLIR_ROOT" \
  --gan_ckpt checkpoints/thermal_gan.pt \
  --diffusion_ckpt checkpoints/object_diffusion/best.pt \
  --detector_ckpt checkpoints/fasterrcnn_flir.pt \
  --profile_path checkpoints/strategy_profile.json \
  --top_k_strategies 3 \
  --split val \
  --num_images 10 \
  --batch_size 1 \
  --context_attack sequential \
  --context_strength 0.12 \
  --max_perturbation 0.20 \
  --n_refinement_steps 8 \
  --device cuda \
  --save_images \
  --out_dir results/profiled_attack
```

Profiled mode uses the stored strategy rankings to narrow the candidate search. Rebuild the profile after changing the model or attack configuration.

## 8. Run the current ablation suites

`run_ablation.py` uses Faster R-CNN guidance, YOLOv11 evaluation, and trained object diffusion. All four checkpoint arguments below are required:

```bash
python run_ablation.py \
  --suite all \
  --data_root "$FLIR_ROOT" \
  --gan_ckpt checkpoints/thermal_gan.pt \
  --diffusion_ckpt checkpoints/object_diffusion/best.pt \
  --detector_ckpt checkpoints/fasterrcnn_flir.pt \
  --yolo_ckpt checkpoints/yolo11_flir.pt \
  --split val \
  --num_images 10 \
  --batch_size 1 \
  --num_workers 4 \
  --device cuda \
  --guidance_scale 0.08 \
  --context_strength 0.12 \
  --max_perturbation 0.20 \
  --n_refinement_steps 8 \
  --asr_denominator all \
  --out_dir results/ablation_yolo11_diffusion
```

Choose `--suite components`, `--suite context`, or `--suite all`:

| Suite      | Variants                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------- |
| components | Clean; Object only; Context only; Object + context; Object + detector; Context + detector |
| context    | None; Contrast; Smoothing; Adaptive cooling; Sequential                                   |

This runner uses fixed cyclic object-class transitions instead of adaptive strategy selection. It prints comparison tables and saves `ablation_results.json`; it does not produce the older five-variant ablation chart. Use `--asr_denominator clean_detected` to report success over eligible clean-correct objects instead of all objects. Its hidden-or-misclassified objective differs from the multi-class runner's strategy-specific success definition.

## 9. Read the outputs

A multi-class attack saves:

```text
results/adaptive_fasterrcnn/
├── attack_results.json
├── attack_0000.png       # only with --save_images
├── attack_0001.png
└── ...
```

`attack_results.json` includes aggregate metrics, per-image records, and strategy statistics. The summary records loop and evaluation detector names, thresholds, object and image ASR, per-class success, detection quality, image quality, and final thermal plausibility measurements.

Object-level ASR divides successful attacked objects by all attacked objects. Success requires a confident clean match and either suppression below the threshold for hiding or a match to the selected target class for misclassification. Image-level ASR counts images with at least one successful object. `overall_success_rate` is the mean of the per-image object success rates, so it can differ from aggregate object-level ASR. Set `--eval_conf` and `--eval_iou` explicitly when comparing runs.

The older `run.py --mode attack` instead writes `eval_summary.json`, `per_image_results.json`, and visualizations under `images/`.
