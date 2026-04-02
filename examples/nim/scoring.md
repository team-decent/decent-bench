# Setup
batch_sizes = [32, 64, 128]
step_sizes = [0.01, 0.001]
local_steps = [1, 5, 10]
comm_probs = [0.2, 0.5, 0.9]

LT-ADMM-VR-2

# Random sampling
## 128 batch 0.001 learning rate
DGD: 14.37%
KGT: 9.86% All sucks
LED: 27.32% 10 local steps
LT-ADMM: 28.91% 10 local steps
LT-ADMM-VR: 28.87% 10 local steps
ProxSkip: 9.65% 0.2 best accuracy, 0.9 best consensus (all sucks)

## 128 batch 0.01 learning rate
DGD: 26.41%
KGT: 9.66% 10 local steps slightly better loss, rest same
LED: 89.49% 10 local steps 
LT-ADMM: 89.25% 10 local steps
LT-ADMM-VR: 89.28% 10 local steps
ProxSkip: 10.43% All similar

## 64 batch 0.001 learning rate
DGD: 9.56%
KGT: 9.56% all same
LED: 38.26% 10 local steps
LT-ADMM: 38.14% 10 local steps
LT-ADMM-VR: 38.12% 10 local steps
ProxSkip: 11.46% 0.9 best consensus too

## 64 batch 0.01 learning rate
DGD: 30.98%
KGT: 11.27% 10 local steps
LED: 90.79% 10 local steps
LT-ADMM: 90.36% 10 local steps
LT-ADMM-VR: 90.38% 10 local steps
ProxSkip: 10.11% 0.9 

## 32 batch 0.001 learning rate
DGD: 9.56%
KGT: 9.56% all same
LED: 38.25% 10 local steps
LT-ADMM: 38.14% 10 local steps
LT-ADMM-VR: 38.11% 10 local steps
ProxSkip: 11.46% 0.9

## 32 batch 0.01 learning rate
DGD: 38.22%
KGT: 9.56% all same, 10 slightly better loss
LED: 89.27% 10 local steps
LT-ADMM: 88.67% 10 local steps
LT-ADMM-VR: 88.76% 10 local steps
ProxSkip: 9.96% 0.2

# Heterogenous sampling (2 samples per partition)
## 128 batch 0.001 learning rate
DGD: 10.47%
KGT: 10.90% 1 local step (all similar)
LED: 36.92% 10 local steps
LT-ADMM: 33.85% 10 local steps
LT-ADMM-VR: 33.82% 10 local steps
ProxSkip: 10.09% 0.2 (all similar)

## 128 batch 0.01 learning rate
DGD: 25.51%
KGT: 10.50% 10 LS
LED: 89.41% 10 LS
LT-ADMM: 89.00% 10 LS
LT-ADMM-VR: 88.66% 10 LS
ProxSkip: 10.07% 0.9

## 64 batch 0.001 learning rate
DNF

## 64 batch 0.01 learning rate
DGD: 34.25%
KGT: 10.65% 10 LS
LED: 90.50% 10 LS
LT-ADMM: 90.21% 10 LS
LT-ADMM-VR: 75.07% 10 LS
ProxSkip: 10.09% 0.9

## 32 batch 0.001 learning rate
DGD: 10.66%
KGT: 9.80% all same
LED: 35.21% 10 LS
LT-ADMM: 34.70% 10 LS
LT-ADMM-VR: 34.96% 10 LS
ProxSkip: 10.44% 0.9

## 32 batch 0.01 learning rate
DGD: 25.61%
KGT: 10.50% 10 LS
LED: 89.53% 10 LS
LT-ADMM: 88.94% 10 LS
LT-ADMM-VR: 58.33% 5 LS
ProxSkip: 10.17% 0.9


# Best settings

## Random:
DGD: 32 batch, 0.01 lr (lr important)
KGT: 64 batch, 0.01 lr, 10 local steps (lr important)
LED: 64 batch, 0.01 lr, 10 local steps (lr important)
LT-ADMM: 64 batch, 0.01 lr, 10 local steps (lr important)
LT-ADMM-VR: 64 batch, 0.01 lr, 10 local steps (lr important)
ProxSkip: All bad, probably needs more time. Seems to want lower LR

## Heterogeneous:
DGD: 64 BS 0.01 LR
KGT: all similar, 10 LS
LED: 64 BS 0.01 LR 10 LS
LT-ADMM: 64 BS 0.01 LR 10 LS
LT-ADMM-VR: 128 BS 0.01 LR 10 LS
ProxSkip: 32 BS 0.001 LR