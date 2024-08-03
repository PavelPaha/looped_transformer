
n_gpu=0

# Linear Regression
#/usr/bin/python3 scripts/train.py --config configs/base_loop.yaml \
#    --wandb.name "LR_baseline_loop2" \
#    --gpu.n_gpu $n_gpu

## Sparse LR
/usr/bin/python3 scripts/train.py --config configs/sparse_LR/base_loop.yaml \
    --wandb.name "SparseLR_baseline" \
    --gpu.n_gpu $n_gpu
#
## Decision Tree
#python scripts/train.py --config configs/decision_tree/base.yaml \
#    --wandb.name "DT_baseline" \
#    --gpu.n_gpu $n_gpu

## ReLU 2NN
#python scripts/train.py --config configs/relu_2nn_regression/base.yaml \
#    --wandb.name "ReLU2NN_baseline" \
#    --gpu.n_gpu $n_gpu
