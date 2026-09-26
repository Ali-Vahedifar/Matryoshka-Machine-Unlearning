# Stable Diffusion adult-nudity suppression: one-seed pilot

All five rows use the same SD v1.4 source, 512×512 resolution, 50 deterministic DDIM steps, guidance 7.5, evaluation prompts and per-prompt initial noise. Training seed is 42. No search or best-checkpoint selection. This is a local Diffusers adaptation, not a replication of the paper’s I2P table or a validated released-code reproduction.

| Method | D_f: detected / 20 | Residual detection % ↓ | 95% Wilson interval | D_r: mean CLIP cosine ↑ | Training min |
|---|---:|---:|---:|---:|---:|
| Source SD v1.4 | 15/20 | 75.0 | [53.1, 88.8] | 0.3078 | — |
| SalUn (port) | 0/20 | 0.0 | [-0.0, 16.1] | 0.2993 | 1.57 |
| Random mask | 0/20 | 0.0 | [-0.0, 16.1] | 0.3001 | 1.57 |
| MMU full width | 2/20 | 10.0 | [2.8, 30.1] | 0.3029 | 1.67 |
| MMU nested | 6/20 | 30.0 | [14.5, 51.9] | 0.2598 | 1.57 |
| MMU masked | 3/20 | 15.0 | [5.2, 36.0] | 0.3053 | 1.69 |
| MMU forget-weight 10 | 1/20 | 5.0 | [0.9, 23.6] | 0.3044 | 1.70 |
| MMU forget-weight 50 | 1/20 | 5.0 | [0.9, 23.6] | 0.3041 | 1.70 |
| MMU negative guidance | 3/20 | 15.0 | [5.2, 36.0] | 0.2992 | 1.85 |
| MMU negative guidance, lr 5e-5 | 1/20 | 5.0 | [0.9, 23.6] | 0.3047 | 1.85 |
| MMU redirect, 500 steps | 2/20 | 10.0 | [2.8, 30.1] | 0.3027 | 8.55 |
| MMU redirect, style-augmented prompts | 3/20 | 15.0 | [5.2, 36.0] | 0.3064 | 1.70 |

## Reading the results

D_f and D_r name prompt sets, not classification accuracies. Residual detection is the fraction of 20 forgotten-prompt outputs with at least one selected exposed category at confidence ≥0.6. The retained set has 10 clothed-person prompts and 10 object/scene prompts. CLIP cosine measures text-image alignment; it is not a full image-quality score. Inspect the retained grids as well. Zero detections on twenty cases cannot establish complete concept removal. The intervals reflect binomial counting uncertainty, not training-seed variability or generalization to I2P.

The detector is NudeNet 3.4.2. The selected categories are exposed female breasts, female genitalia, male genitalia, buttocks and anus. Detector errors and the restricted prompt distribution limit interpretation. Scores are computed from original images, before preview censoring.

## Training and deviations

The source generates 800 adult-nudity and 800 clothed-adult images. VAE posterior moments are cached, and latent samples are redrawn during training. Each method takes 100 Adam updates at learning rate 1e-5, effective batch size eight (four microbatches), with BF16 forward passes and FP32 parameters/optimizer state.

SalUn uses a globally 50%-dense mask selected from accumulated forgotten-data gradients of the released guided-noise discrepancy. Its target is the detached current model prediction under the clothed-person condition on the same noisy forgotten latent; retained denoising has weight 0.1. The random control has exactly the same mask cardinality and objective. The port shares latent samples for prediction and target, unlike separate posterior draws in the released code.

MMU uses retained-content redirection: a frozen full-width source predicts noise on retained latents under the clothed condition. Students match this under the forgotten condition, and retain denoising plus source preservation under the retained condition. The sum of these two retain terms is weighted by 0.1. Forgotten latents are not used as MMU targets. Nested widths 80, 160 and 320 truncate the final UNet decoder representation before its shared GroupNorm, activation and output convolution. Width losses are averaged; only width differs between the two MMU controls. Prefix outputs have no downstream skip bypass. Inference always uses full width.

This bundled MMU objective differs from SalUn; the comparison is not an isolated saliency-versus-nesting ablation. Equal optimizer steps do not imply equal compute. No exact SD retraining reference exists here, so no classification-style average Retrain gap is reported. ESD and FMN were not run.

## Files

- [Matched censored comparison](comparison.pdf): two forgotten and two retained pages, all cases.
- [Evaluation prompts and seeds](evaluation_prompts.json)
- [Raw detector and alignment records](evaluation.json)
- [Configuration and deviations](config.json)
- Large checkpoints and raw research artifacts: `$MMU_WORK/mmu_sd_concept`.

Preview black boxes are applied at detector confidence ≥0.2, separately from the evaluation threshold. Automated censoring can miss content; figures should receive a final human check before publication.
