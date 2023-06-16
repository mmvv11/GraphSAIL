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
python model.py
```
