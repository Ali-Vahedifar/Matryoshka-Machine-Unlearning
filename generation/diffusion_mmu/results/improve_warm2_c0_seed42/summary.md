# Diffusion evaluation: guidance 2.0

Forget class 0; seed 42; 500 forgotten-condition samples; 5000 retained samples.

| Method | UA (%) | Retained FID | Retained accuracy (%) | Train/load time (s) |
|---|---:|---:|---:|---:|
| MMU full width | 85.2 | 17.05 | 96.9 | 60.9 |
| MMU nested | 72.4 | 19.47 | 95.9 | 118.7 |

UA alone cannot distinguish valid redirection from damaged generation.
SalUn is a local adaptation, not a validated reproduction. One seed and one class do not establish general superiority.
