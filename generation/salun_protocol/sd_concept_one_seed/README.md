# Stable Diffusion nudity-suppression pilot

Status: implementation complete and one-seed run in progress. See [live status](status.json), [fixed configuration](config.json), and `$MMU_WORK/mmu_sd_concept/run.log`. Training data generation is underway; no final result is claimed until evaluation completes.

Purpose: evaluate whether MMU suppresses adult nudity while preserving ordinary
image generation. This is a new Stable Diffusion adaptation, not the existing
CIFAR-10 DDPM experiment or an already validated MMU result.

## Fixed scope

- One training seed (42); no hyperparameter sweep or ten-trial study.
- Shared Stable Diffusion v1.4 source for all methods.
- Comparison rows: source SD, released-code SalUn, a matched random-mask
  control, an MMU full-width control and nested MMU.
- Match training examples, evaluation prompts, initial noise, image size,
  scheduler, guidance and inference steps across methods. Record update
  counts and runtime rather than implying equal compute from equal epochs.
- Preserve all outcomes. Report figures use censored previews where needed;
  suppression metrics are computed before presentation censoring.
- ESD and FMN require their own validated implementations/checkpoints before
  adding those labels; they are not silently substituted by local variants.

## Released SalUn protocol to reproduce first

The pinned local SD README prescribes SD v1.4, 800 generated forgotten images
and 800 clothed-person retained images, a saliency mask, and the released
`train-scripts/nsfw_removal.py` entry point. Its evaluation script uses NudeNet.
The exact detector version, category mapping, threshold, data-generation seeds
and evaluation prompts must be pinned before reporting a reproduction.
Review evaluation prompts for adult-only, non-exploitative research scope.

Sources: `../salun_official/SD/README.md` relative to `salun_protocol`'s parent;
more directly, `salun_official/SD/README.md`,
`SD/train-scripts/nsfw_removal.py` and `SD/eval-scripts/nudenet-classes.py`.
Official documentation:
https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/SD

## MMU implementation design

Select and document the UNet representation where prefixes operate; check
skip connections and retained prefix prediction quality. Use the full-width
frozen teacher as the retained target. Compare nested and full-width controls
with the same objective and averaged width weighting. A classification
soft-target loss cannot simply be copied into a diffusion denoiser. Validate
finite gradients, detached targets, fixed teacher buffers and checkpoint
round trips before launching the one-seed run.

## Evaluation

- Forgotten prompt set: image-level residual nudity-detection rate and counts,
  with a fixed detector version, category set and threshold. Lower is preferred.
- Retained prompts: prompt alignment and image quality, accompanied by matched
  sample grids. Include clothed people and unrelated objects/scenes to detect
  broad damage. Do not compute or advertise a reliable FID from ten previews.
- Keep the same prompt/seed in each figure column. Record prompt IDs and
  disclose all selected columns; do not select only successful MMU outputs.
- D_f and D_r refer to forgotten and retained sets. Classification accuracy
  metrics do not directly carry over to text-conditioned concept removal.
- Exact retraining of SD without the concept is unavailable in this pilot;
  do not invent a Retrain row or a classification-style average Retrain gap.

## Environment

Large artifacts (the SD v1.4 Diffusers checkpoint, generated training data, method
checkpoints) go to `$MMU_WORK/mmu_sd_concept`; allow about 30 GB. The objective is
ported to Diffusers rather than the legacy CompVis runtime, as recorded in `config.json`.

The implemented pilot uses 20 adult-only nonsexual art/figure prompts and 20 retained prompts, one image per prompt per method. It is not the original I2P evaluation. Full-width and nested MMU use retained-content redirection, with prefixes at the final UNet decoder feature map. Detailed settings and deviations are fixed in config.json.
