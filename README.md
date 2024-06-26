# SLTrain
A repository containing beta implementation for *SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining*. Preprint available on http://arxiv.org/abs/2406.02214.

## Modeling for pretraining
W = BA + S, 

where B, A model the low-rank component and S models the sparse component. S has a **random** sparsity pattern.

## Motivation
Below, we show how the learned weights L + S enlarges the spectrum. In particular, the L component primarily learns the head singular value spectrum and the S component primarily learns the tail spectrum. 

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/SLTrain_fig1.png?raw=true" alt="Contribution of L and S components in the singular values of learned W" width="1000" height="600">

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/SLTrain_fig2.png?raw=true" alt="Zoomed view" width="450" height="400">


## Results
SLtrain offers a good trade-off between perplexity, param, and memory requirements.

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/SLTrain_fig3.png?raw=true" alt="Perp, param, memory trade-off">

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/SLTrain_fig4.png?raw=true" alt="7B memory saving">
