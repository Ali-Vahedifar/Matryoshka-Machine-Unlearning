# Machine unlearning baselines

One folder per method. Each exposes a class with `unlearn(...)`, and `common.py` holds
the shared helpers (distillation KL, frozen copies, Fisher, retain training). MMU itself
is in `../MMU/`.

| Folder | Method | Reference |
|---|---|---|
| `Baseline` | Original model, no unlearning | — |
| `Retrain` | retrain from scratch on D_r | — |
| `FineTune` | fine-tuning on the accessible retain set | — |
| `BadTeacher` | competent/incompetent-teacher distillation | Chundawat et al., AAAI 2023 |
| `Amnesiac` | subtracts the updates recorded for forget batches during training (class deletion only) | Graves et al., AAAI 2021 |
| `UNSIR` | error-maximising noise, impair then repair | Tarun et al., 2023 |
| `SSD` | selective synaptic dampening (Eq. 4) | Foster et al., AAAI 2024 |
| `SalUn` | saliency mask on forget gradients + random labels | Fan et al., ICLR 2024 |
| `UniCLUN` | bounded replay, dual-teacher distillation | Chatterjee et al., 2024 |
| `SCRUB` | teacher–student max/min steps | Kurmanji et al., NeurIPS 2023 |

RL, GA, IU, ℓ1-sparse, BS and BE come from the official SalUn release in
`salun_official/Classification/unlearn/`, driven by `generation/salun_protocol/classification.py`.
