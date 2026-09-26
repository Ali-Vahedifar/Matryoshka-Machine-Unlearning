# Diffusion evaluation: guidance 2.0

Forget class 0; seed 43; 500 forgotten-condition samples; 5000 retained samples.

| Method | UA (%) | Retained FID | Retained accuracy (%) | Train/load time (s) |
|---|---:|---:|---:|---:|
| MMU full width | 92.2 | 17.01 | 97.7 | 39.8 |

UA alone cannot distinguish valid redirection from damaged generation.
SalUn is a local adaptation, not a validated reproduction. One seed and one class do not establish general superiority.
