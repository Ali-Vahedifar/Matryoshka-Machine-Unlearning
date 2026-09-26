# Diffusion evaluation: guidance 2.0

Forget class 0; seed 42; 500 forgotten-condition samples; 5000 retained samples.

| Method | UA (%) | Retained FID | Retained accuracy (%) | Train/load time (s) |
|---|---:|---:|---:|---:|
| Source | 1.4 | 16.02 | 98.2 | 0.1 |
| Retrain reference | 84.0 | 15.91 | 97.9 | 0.1 |
| Fine-tune | 0.6 | 17.58 | 97.3 | 10.5 |
| NegGrad | 100.0 | 304.80 | 5.6 | 20.7 |
| SalUn (local adaptation) | 98.4 | 18.59 | 96.4 | 26.4 |
| MMU full width | 86.2 | 17.66 | 97.6 | 40.6 |
| MMU nested | 49.0 | 25.72 | 93.7 | 83.8 |
| MMU untruncated skips | 85.2 | 17.39 | 97.7 | 83.4 |
| MMU retain only | 4.4 | 24.95 | 94.8 | 52.1 |

UA alone cannot distinguish valid redirection from damaged generation.
SalUn is a local adaptation, not a validated reproduction. One seed and one class do not establish general superiority.
