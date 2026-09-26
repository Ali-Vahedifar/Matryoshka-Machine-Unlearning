# Diffusion evaluation: guidance 1.0

Forget class 0; seed 42; 500 forgotten-condition samples; 5000 retained samples.

| Method | UA (%) | Retained FID | Retained accuracy (%) | Train/load time (s) |
|---|---:|---:|---:|---:|
| Source | 12.8 | 22.05 | 81.5 | 0.1 |
| Retrain reference | 93.6 | 21.67 | 82.4 | 0.1 |
| Fine-tune | 10.8 | 27.39 | 78.7 | 10.5 |
| NegGrad | 100.0 | 304.34 | 15.3 | 20.7 |
| SalUn (local adaptation) | 97.6 | 31.11 | 76.8 | 26.4 |
| MMU full width | 90.8 | 24.40 | 79.1 | 40.6 |
| MMU nested | 81.4 | 34.97 | 69.0 | 83.8 |
| MMU untruncated skips | 91.8 | 23.14 | 80.6 | 83.4 |
| MMU retain only | 30.4 | 34.54 | 72.1 | 52.1 |

UA alone cannot distinguish valid redirection from damaged generation.
SalUn is a local adaptation, not a validated reproduction. One seed and one class do not establish general superiority.
