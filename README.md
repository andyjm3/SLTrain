# SLTrain
A repository containing beta implementation for *SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining*, **which has been accepted to NeurIPS 2024**. Preprint available on http://arxiv.org/abs/2406.02214.

## Modeling for pretraining
The main idea is to re-parameterize linear layer with low-rank and sparse factors for improved parameter and memory efficiency.


W = BA + S, 

where B, A model the low-rank component and S models the sparse component. S has a **random** sparsity pattern.

## Motivation
Below, we show how the learned weights L + S enlarges the spectrum. In particular, the L component primarily learns the head singular value spectrum and the S component primarily learns the tail spectrum. 

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/SLTrain_fig1.png?raw=true" alt="Contribution of L and S components in the singular values of learned W" width="1000" height="600">

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/SLTrain_fig2.png?raw=true" alt="Zoomed view" width="450" height="400">

## Results

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/sltrain_result_all.png?raw=true" alt="Result Comparisons" width="500" height="500">

<img src="https://github.com/andyjm3/SLTrain/blob/main/figures/sltrain_result_memory.png?raw=true" alt="SlTrain Memory" width="500" height="500">

## Installation

Build cpp extensions via
```bash 
cd ./sparse-lora
pip install .
```

## Usage

Run the scripts placed in scripts/llm_pretrain/. Typical usage:

```bash
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.003 \
    --peft_model sltrain\
    --optimizer adamw \
    --rank 128 \  
    --sp_ratio 0.03 \  # sparsity delta
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 32 
```

## Citation
```bibtex
@article{han2024sltrain,
  title={{SLTrain}: a sparse plus low-rank approach for parameter and memory efficient pretraining},
  author={Han, Andi and Li, Jiaxiang and Huang, Wei and Hong, Mingyi and Takeda, Akiko and Jawanpuria, Pratik and Mishra, Bamdev},
  journal={arXiv preprint arXiv:2406.02214},
  year={2024}
}
```
