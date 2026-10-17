# Algorithm test layout

- `p2p/` contains peer-to-peer algorithm reference tests, organized by update family.
- `federated/` contains federated algorithm and protocol tests.
- `shared/` contains smoke tests that apply to both families.

The shared smoke matrix checks construction, execution with and without network impairments, and output shape and
finiteness for every algorithm listed in its P2P and federated case tables. It is an integration check, not a substitute
for update-rule references.

## P2P reference coverage

| Algorithm | Current reference check |
| --- | --- |
| DGD, ATC | One update |
| SimpleGT, ED, EXTRA, ATC_Tracking, AugDGM, NIDS, WangElia, KGT, LED, ProxSkip | Two updates, including the carried tracker or auxiliary state |
| ADMM, ATG, DiNNO, LT_ADMM | Two updates, including edge variables or dual state |
| DLM | Three iterations, including its initialization communication round |
| GT_SAGA, GT_SARAH, GT_VR, LT_ADMM_VR | Smoke coverage only; add family-specific reference cases |

For tracking algorithms, a reference case must include at least two updates so it checks the recurrence after tracker
initialization. Stochastic methods should use seeded randomness or assert a documented invariant rather than compare an
unseeded trajectory.
