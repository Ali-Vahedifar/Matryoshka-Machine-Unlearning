# MMU variant names

| ID | Explicit name | Privileged target |
|---|---|---|
| V01 | MMU unbounded / full | width-aligned source |
| V02 | MMU unbounded / nested MRL / sum | width-aligned source |
| V03 | MMU unbounded / nested MRL / weighted mean | width-aligned source |
| V04 | MMU unbounded / nested MRL-E / sum | width-aligned source |
| V05 | MMU unbounded / nested MRL-E / weighted mean | width-aligned source |
| V06 | MMU margin / full | width-aligned source |
| V07 | MMU margin / nested MRL / sum | width-aligned source |
| V08 | MMU margin / nested MRL / weighted mean | width-aligned source |
| V09 | MMU margin / nested MRL-E / sum | width-aligned source |
| V10 | MMU margin / nested MRL-E / weighted mean | width-aligned source |
| V11 | MMU margin / nested MRL / full privileged teacher | full source |
| V12 | MMU retain-FT two-network / full / attraction only | full retain-FT + full source |
| V13 | MMU retain-FT two-network / full / attraction + margin | full retain-FT + full source |
| V14 | MMU retain-FT two-network / nested MRL / attraction only | full retain-FT + full source |
| V15 | MMU retain-FT two-network / nested MRL / attraction + margin | full retain-FT + full source |

Every variant uses frozen-teacher distillation. Source-role variants use one source network on retained and forgotten data; retain-FT adaptations use two distinct networks. Nested means prefixes [32,64,128,256,512]. MRL uses separate heads; MRL-E ties the head. Full means no nesting. Weighted mean uses the original normalised per-width weighting, not a simple equal average. No moving-student self-distillation is included, consistent with your clarified definition.

V11 is the combination with nested student prefixes, full privileged source targets, and margin-limited two-role source distillation. V02–V10 instead use width-aligned source targets when nested. V12–V15 are the explicitly separate retain-FT classification adaptations; they must not be presented as identical to conditioned diffusion MMU.

Class: airplane removal, 10 labels. Instance: 4,500 training examples. Subclass: airplane removal from the airplane/ship superclass in a custom 5-way task. Backbone: ResNet-18; training seeds 42,43,44. Source/Retrain caches are reused only after protocol/hash checks. All unlearning runs are new. Fixed-margin sensitivity and historical learning-rate candidates are not counted as additional structural methods.

The final report will be generated after the runs and full training-subset evaluations finish. Results/logs/checkpoints are retained in $MMU_WORK/mmu_variants_3seed.