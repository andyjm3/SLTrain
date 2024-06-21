# SLTrain
A repository containing beta implementation for *SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining*. Preprint available on http://arxiv.org/abs/2406.02214.

## Modeling for pretraining
W = BA + S, 

where B, A model the low-rank component and S models the sparse component. S has a **random** sparsity pattern.

## Motivation
Below, we show how the learned weights L + S enlarges the spectrum. In particular, the L component primarily learns the head singular value spectrum and the S component primarily learns the tail spectrum. 
![alt text](https://www.dropbox.com/scl/fi/qhw27ed4f96qik0ac1hk1/SLTrain_fig1.png?rlkey=1c6broxid2pdpidbg0cgok743&st=nm2ojhc5&dl=1)
![alt text](https://www.dropbox.com/scl/fi/ceow5bywvauyp15kz9kdf/SLTrain_fig2.png?rlkey=isucat61bbl1hlnktsk3zofsh&st=cx0oe4h9&dl=0)


