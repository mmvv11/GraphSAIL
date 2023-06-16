# PinSAGE example

## Requirements

- dask
- pandas

## Prepare datasets

### MovieLens 1M

1. Download and extract the MovieLens-1M dataset from http://files.grouplens.org/datasets/movielens/ml-1m.zip
   into `./dataset/ml-1m`.
2. Run `python graphsail_process_movielens1m.py ./dataset/ml-1m`
   Replace `ml-1m` with the directory you put the `.dat` files, and replace `data_processed` with
   any path you wish to put the output files.

## Run model

### Nearest-neighbor recommendation

This model returns items that are K nearest neighbors of the latest item the user has
interacted.  The distance between two items are measured by Euclidean distance of
item embeddings, which are learned as outputs of PinSAGE.

```
python model.py data_processed --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 64
```

The implementation here also assigns a learnable vector to each item.  If your hidden
state size is so large that the learnable vectors cannot fit into GPU, use this script
for sparse embedding update (written with `torch.optim.SparseAdam`) instead:


```
python model_sparse.py data_processed --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 1024
```
