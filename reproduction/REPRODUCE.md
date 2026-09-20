# Reproduction materials

- Data-generation script: `generate_data.py`.
- Configurations and semantic splits: `experiments.json`.

## Data generation

Install upstream dependencies and generator checkpoints.

Run from the SITraNet root. `REPO` is relative to this root. Input/output paths are relative to the upstream repository; mas_GRDH uses its `scripts/` directory.

```bash
python reproduction/generate_data.py METHOD REPO [upstream arguments]

python reproduction/generate_data.py IDEAS third_party/IDEAS \
  --checkpoint checkpoints/model.pt --outdir outputs/stego --count 20000 --seed 42

python reproduction/generate_data.py StyleGAN2 third_party/stylegan2 generate-images \
  --network=checkpoints/model.pkl --seeds=0-19999 --truncation-psi=1.0 --result-dir=outputs/covers
```

| Method | Upstream instructions | Called entry |
| --- | --- | --- |
| IDEAS | [IDEAS](https://github.com/Lemok00/IDEAS) | `models.init_model`, `utils.message_to_tensor` |
| SI-SWE | [SI-SWE](https://github.com/Lemok00/SI-SWE) | `synthesise_images_and_calculate_fid.py` |
| StyleGAN2 | [StyleGAN2](https://github.com/NVlabs/stylegan2) | `run_generator.py generate-images` |
| CRoSS | [CRoSS](https://github.com/yujiwen/CRoSS) | `demo.py` |
| DiffStega | [DiffStega](https://github.com/evtricks/DiffStega) | `main.py` |
| mas_GRDH | [mas_GRDH](https://github.com/HXX5656/mas_GRDH) | `scripts/txt2img.py` |

- SI-SWE: use the original release and its checkpoints. Follow its README for generation arguments.
- mas_GRDH: enable PNG storage; one prompt and separate output folder per call.
- Keep generated stegos only; CRoSS uses `hide.png`. Record prompts, inputs and weights.
- IDEAS/SI-SWE: 256-bit messages. Diffusion methods: native payloads.
- Inputs: 256×256 RGB. Cover sources: Section 5.1; training excludes StyleGAN/StyleGAN2, testing uses matched StyleGAN2.

## Apply configurations

- Read `protocols`, `shared_settings` and `new_experiments` in JSON.
- Set the parameters in `train.py` and `test.py` according to `experiments.json`.
- `train.py`: set `TRAIN_DOMAINS`, `TRAIN_COVER_DIRS`; both dataset calls use `samples_per_domain=20000`.
- Source train/validation: 9:1. Checkpoint: highest validation ACC; save condition `if val_acc > best_acc:`.
- Run `SITRANET_SEED=42 python train.py`; repeat for the other configured seeds.
- `test.py`: set checkpoint, stego/cover paths and `TEST_SAMPLES=2000`. Use unique, fixed test IDs across models.
- SITraNet encoders: ImageNet initialization. Baselines: trained from scratch.
- Other training settings: unchanged from the specified protocol.

## Generator-controlled experiment

- Protocol 1B; SITraNet and FMISNet.
- Training covers: matched StyleGAN2 for CelebA, Church and Bedroom.
- Retrain with each cover setting. Keep target data fixed.
- Output: ACC for each semantic domain and average ACC.

## Diffusion cross-method experiment

- Sources: CRoSS, mas_GRDH; joint semantics Bridge, Bedroom, Face.
- Target: DiffStega; Bedroom, Church, CelebA.
- SITraNet and FMISNet: compare direct GAN transfer with diffusion-source retraining.
- Select by validation ACC. Report ACC for each target domain and average ACC.

## Post-processing robustness

- Protocol 1A; SITraNet, FMISNet, SiaStegNet. Frozen checkpoints.
- Conditions: original images; JPEG 95/75; Scale 0.50; Crop 0.80.
- Apply each operation separately to cover and stego images before normalization.
- Resize/crop: downsample or center crop; restore to 256×256 with bicubic interpolation.
- Output: ACC for each semantic domain and average ACC.

## Complementary metrics

- Protocols 1A/1B; SITraNet, FMISNet, SiaStegNet. Five independent training runs.
- Save labels, predictions and Stego probabilities for fixed test IDs.
- Metrics: ACC, AUC, EER, TPR at 1% FPR, FPR at 95% TPR.
- ACC = 0.5 × (TP/P + TN/N) × 100%. Stego is the positive class.
- Average the six target-domain results in Protocol 1 for each run.
- Report ACC as mean ± standard deviation over five independent runs.
- Compute other metrics for each domain, then report their average.

## Encoder backbones

- Protocol 1A; Xception, ResNet-18, EfficientNet. ImageNet initialization.
- Use the selected backbone for both the trace and semantic encoders in `sitranet.py`.
- Match decoder/fusion input channels. Retrain each variant; retain other settings.
- Output: ACC for each semantic domain and average ACC.

## Computational efficiency

- SITraNet, FMISNet, SiaStegNet; RTX 3090; FP32; batch size 1.
- Evaluation mode; no gradients. Warm up and synchronize CUDA around timing.
- Output: operations and inference time per image. Exclude loading and image I/O.
- K: 1, 2, 4, 8, 16. Match prototype and distance-head sizes.

## Semantic leakage

- Frozen features; identical Bedroom/CelebA/Church stegos.
- Compare SITraNet 256-D trace, independently trained non-disentangled variant, FMISNet and Xception.
- Train a logistic-regression classifier to predict the semantic domain.
- Use the same training/test split for all features. Report semantic classification ACC.
