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

# LT-ADMM-EMA

## Random
Torch Optimizer:
| Alg | Consensus | Gradients | Accuracy | Loss |
|-|-|-|-|-|
| EMA-OPT-LS20-SETrue_ef_0.5_ze_True_bs_32_ss_0.01 | 0.008725030161440373 | 640000.0 | 0.94856 | 3.2396150778453146e-07 |
| EMA-OPT-LS10-SEFalse_ef_0.5_ze_True_bs_32_ss_0.01 | 0.01254148967564106 | 320000.0 | 0.94764 | 5.240680243878159e-06 |
| EMA-OPT-LS20-SETrue_ef_0.5_ze_False_bs_32_ss_0.01 | 0.007471023034304381 | 640000.0 | 0.9472 | 1.1677716209277379e-07 |
| EMA-OPT-LS20-SETrue_ef_0.9_ze_False_bs_32_ss_0.01 | 1.4211519956588745 | 640000.0 | 0.9468 | 0.0012883220135408008 |
| EMA-OPT-LS10-SEFalse_ef_0.1_ze_True_bs_32_ss_0.01 | 0.01215035207569599 | 320000.0 | 0.94672 | 3.5286862019972886e-06 |
| EMA-OPT-LS15-SETrue_ef_0.5_ze_True_bs_32_ss_0.01 | 0.013820537365972996 | 480000.0 | 0.94668 | 2.224581290309402e-06 |

Alg optimizer:
| Alg | Consensus | Gradients | Accuracy | Loss |
|-|-|-|-|-|
| EMA-LS20-SEFalse_ef_0.5_ze_False_bs_32_ss_0.01 | 0.01605018675327301 | 640000.0 | 0.9178 | 0.06792903148382902 | 
| EMA-LS20-SEFalse_ef_0.9_ze_False_bs_32_ss_0.01 | 0.0029499959666281937 | 640000.0 | 0.91744 | 0.06761873082518578 | 
| EMA-LS20-SEFalse_ef_0.1_ze_True_bs_32_ss_0.01 | 0.03222256377339363 | 640000.0 | 0.916559 | 0.0868315397694707 | 
| EMA-LS20-SETrue_ef_0.1_ze_True_bs_32_ss_0.01 | 0.03168455176055431 | 640000.0 | 0.91536 | 0.09881225630044936 | 
| EMA-LS15-SEFalse_ef_0.9_ze_False_bs_32_ss_0.01 | 0.0032192666549235582 | 480000.0 | 0.914 | 0.12521634695529937 | 
| EMA-LS15-SEFalse_ef_0.5_ze_False_bs_32_ss_0.01 | 0.01572207026183605 | 480000.0 | 0.9134 | 0.1235556604474783 | 

## Heterogeneous
Torch Optimizer:
| Alg | Consensus | Gradients | Accuracy | Loss |
|-|-|-|-|-|
| 	EMA-OPT-LS15-SEFalse_ef_0.9_ze_False_bs_32_ss_0.005|	0.5746125459671021	|480000.0	|0.83952	|0.2796987950004637|
| 	EMA-OPT-LS10-SEFalse_ef_0.9_ze_False_bs_32_ss_0.01	|0.82070974111557	|320000.0	|0.748719	|0.8583789972894593|
| 	EMA-OPT-LS15-SEFalse_ef_0.5_ze_False_bs_32_ss_0.005	|2.0921759366989137	|480000.0	|0.69056	|0.0027204027424682863|
| 	EMA-OPT-LS15-SEFalse_ef_0.9_ze_False_bs_32_ss_0.01	|0.9005046725273133	|480000.0	|0.6724	|1.6554778213858605|
| 	EMA-OPT-LS10-SEFalse_ef_0.5_ze_False_bs_32_ss_0.01	|2.8763850688934327	|320000.0	|0.6242	|0.010617288185728829|
| EMA-OPT-LS20-SEFalse_ef_0.1_ze_True_bs_32_ss_0.005	|3.260079002380371	|640000.0	|0.6228	|0.004558282169525046|

Alg optimizer:
| Alg | Consensus | Gradients | Accuracy | Loss |
|-|-|-|-|-|
| EMA-LS15-SEFalse_ef_0.5_ze_False_bs_32_ss_0.01 | 	0.01582234725356102 |	480000.0	|0.91648	|0.11874181624799966|
| EMA-LS15-SEFalse_ef_0.9_ze_False_bs_32_ss_0.01 | 	0.003121757227927446 |	480000.0	|0.9164	|0.12040784177184105|
| EMA-LS15-SEFalse_ef_0.1_ze_True_bs_32_ss_0.01	 | 0.028187885135412215	| 480000.0	|0.91592	|0.12903903867900374|
| EMA-LS20-SEFalse_ef_0.5_ze_False_bs_32_ss_0.01 | 	0.017426000349223612 | 640000.0	|0.909639	|0.11494036626145243|
| EMA-LS20-SEFalse_ef_0.9_ze_False_bs_32_ss_0.01 | 	0.004517911607399583 | 640000.0	|0.90936	|0.1201397696748376|
| EMA-LS20-SEFalse_ef_0.1_ze_True_bs_32_ss_0.01	 | 0.03781130686402321	| 640000.0	|0.90696	|0.12758788799345494|


# Final experiments - Random
## DGD
94.6% ss1: 0.5, ss2: 1.0, bs: 64
94.2% ss1: 0.5, ss2: 1.0, bs: 32
91.9% ss1: 0.1, ss2: 1.0, bs: 64
91.5% ss1: 0.1, ss2: 1.0, bs: 32

## KGT
consensus: 1e-7
91.6% ls: 15, ss1: 0.01, ss2: 0.5, bs: 64
91.2% ls: 15, ss1: 0.01, ss2: 0.5, bs: 32
90.7% ls: 10, ss1: 0.01, ss2: 0.5, bs: 64
90.4% ls: 10, ss1: 0.01, ss2: 0.5, bs: 32

## LED
93.6% ls: 15, ss1: 0.01, ss2: 0.01, bs: 64
93.5% ls: 15, ss1: 0.01, ss2: 0.1, bs: 64
93.4% ls: 15, ss1: 0.01, ss2: 0.001, bs: 64
93% ls: 15, ss1: 0.01, ss2: 0.01, bs: 32

## ProxSkip
95.2% ss1: 0.5, ss2: 0.001, cp: 1.0, chi: 1.5, bs: 64
95.1% ss1: 0.5, ss2: 0.1, cp: 1.0, chi: 2.0, bs: 64
95% ss1: 0.5, ss2: 0.01, cp: 1.0, chi: 1.0, bs: 64
94.9% ss1: 0.5, ss2: 0.01, cp: 1.0, chi: 2.0, bs: 64

## LT-ADMM
92.8% ls: 15, ss1: 0.01, ss2: 0.01, p: 0.5, bs: 64
92.5% ls: 15, ss1: 001, ss2: 0.01, p: 0.5, bs: 32
92.5% ls: 15, ss1: 0.01, ss2: 0.001, p: 0.5, bs: 32
92.5% ls: 15, ss1: 0.01, ss2: 0.001, p: 1.0, bs: 32

## LT-ADMM-EMA
92.4% ls: 15, ss1 0.01, ss2: 0.001, p: 0.5, ema: 0.5, bs: 32
92.3% ls: 15, ss1: 0.01, ss2: 0.01, p: 0.5, ema: 0.9, bs: 32
92.2% ls: 15, ss1: 0.01, ss2: 0.001, p: 0.5, ema: 0.7, bs: 32
92.2% ls: 15, ss1: 0.01, ss2: 0.01, p: 0.5, ema: 0.5, bs: 32

## LT-ADMM-EMA-OPT
95.7% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.0, ema: 0.5, bs: 32
95.6% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.5, ema: 0.5, bs: 32
95.6% ls: 10, ss1: 0.01, ss2: 0.01, p: 1.0, ema: 0.9, bs: 32
95.6% ls: 15, ss1: 0.01, ss2: 001, p: 1.0, ema: 0.7, bs: 32

# Final experiments - hetero
## DGD
90% ss1: 0.1, ss2: 1.0, bs: 64
81% ss1: 0.5, ss2: 1.0, bs: 64
79% ss1: 0.01, ss2: 1.0, bs: 64
75% ss1: 0.2, ss2: 1.0, bs: 32

## KGT
91% ls: 15, ss1: 0.01, ss2: 0.5, bs: 32
01% ls: 15, ss1: 0.01, ss2: 0.5, bs: 64
90% ls: 10, ss1: 0.01, ss2: 0.5, bd: 32
90% ls: 10, ss1: 0.01, ss2: 0.5, bs: 64

## LED
93% ls: 15, ss1: 0.01, ss2: 0.1, bs: 64
93% ls: 15, ss1: 0.01, ss2: 0.1, bs: 32
93% ls: 15, ss1: 0.01, ss2: 0.01, bs: 64
92% ls: 15, ss1: 0.01, ss2: 0.01, bs: 32

## ProxSkip
95% ss1: 0.5, ss2: 0.5, cp: 1.0, chi: 1.0, bs: 32
94% ss1: 0.5, ss2: 0.5, cp: 1.0, chi: 1.0, bs: 64
94% ss1: 0.5, ss2: 0.1, cp: 1.0, chi: 1.0, bs: 64
93% ss1: 0.5, ss2: 0.5, cp: 1.0, chi: 1.5, bs: 64

## LT-ADMM
93% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.5, bs: 64
93% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.5, bs: 32
92% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.0, bs: 32
92% ls: 10, ss1: 0.01, ss2: 0.01, p: 1.5, bs: 32

## LT-ADMM-EMA
93% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.5, ema: 0.9, bs: 32
93% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.5, ema: 0.5, bs: 32
93% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.5, ema: 0.7, bs: 32
92% ls: 15, ss1: 0.01, ss2: 0.01, p: 1.0, ema: 0.5, bs: 32

## LT-ADMM-EMA-OPT
87% ls: 10, ss1: 0.001, ss2: 0.01, p: 1.0, ema: 0.7, bs: 32
86% ls: 15, ss1: 0.001, ss2: 0.01, p: 1.5, ema: 0.7, bs: 32
86% ls: 10, ss1: 0.001, ss2: 0.01, p: 1.5, ema: 0.5, bs: 32
85% ls: 10, ss1: 0.001, ss2: 0.01, p: 1.0, ema: 0.5, bs: 32