# SLTrain
A repository containing beta implementation for *SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining*. Preprint available on http://arxiv.org/abs/2406.02214.

## Modeling for pretraining
W = BA + S, 

where B, A model the low-rank component and S models the sparse component. S has a **random** sparsity pattern.

## Motivation
Below, we show how the learned weights L + S enlarges the spectrum. In particular, the L component primarily learns the head singular value spectrum and the S component primarily learns the tail spectrum. 

![alt text](https://github.com/bamdevm/SLTrain/blob/patch-2/figures/SLTrain_fig1.png?raw=true)
![alt text](https://github.com/bamdevm/SLTrain/blob/patch-2/figures/SLTrain_fig2.png?raw=true)



