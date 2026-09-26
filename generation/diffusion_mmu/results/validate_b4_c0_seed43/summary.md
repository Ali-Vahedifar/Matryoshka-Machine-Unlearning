# Diffusion evaluation: guidance 2.0

Forget class 0; seed 43; 500 forgotten-condition samples; 5000 retained samples.

| Method | UA (%) | Retained FID | Retained accuracy (%) | Train/load time (s) |
|---|---:|---:|---:|---:|
| MMU full width | 91.2 | 19.60 | 97.0 | 40.9 |

UA alone cannot distinguish valid redirection from damaged generation.
SalUn is a local adaptation, not a validated reproduction. One seed and one class do not establish general superiority.
