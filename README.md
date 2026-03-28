# antibiotic_benchmark

## Setups

```bash
# 0) clone with submodules
git clone --recurse-submodules git@github.com:MolAstra/antibiotic_benchmark.git
cd antibiotic_benchmark

# if repo already exists locally, run this once instead:
# git submodule sync --recursive
# git submodule update --init --recursive

# create env
mamba create -n antibio_bench python=3.12
mamba activate antibio_bench

# install pytroch packages
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu126.html

# install dpnet for data spliting
pip install "git+https://github.com/zhaisilong/dpnet.git"

# install antibio_bench package
pip install -e .
# matplotlib is installed automatically via pyproject dependencies

# optional: keep submodules updated after pull
# git pull --recurse-submodules
# git submodule update --init --recursive
```

## Data Preparation and Standardization

`0_data_preparation.ipynb` prepares and standardizes the input table, then exports genus-level datasets used by downstream DPNet split and modeling scripts.

## Data Splitting (DPNet)

Use `1_run_dpnet.sh` to build per-genus DPNet tasks and run DPNet scaffold splitting.

- Input: `datasets/genus_top15/*.csv`
- DPNet task root: `datasets/dpnet_db/<genus>/`
- Split output: `datasets/dpnet_db/<genus>/processed/<genus>/{train,valid,test}.csv`
- Ratio summary output: `datasets/data_pos_ratio.csv`

```bash
bash 1_run_dpnet.sh
```

## Modeling

`run_ml_single.py` / `run_ml_single.sh` train one model for one dataset + one cutoff using pre-split DPNet files (`train/valid/test`).

`run_ml_batch.sh` runs batch experiments across all processed datasets, model list, and cutoff list.

- Models: `svm`, `rf`, `lightgbm`, `xgb`
- Output: `resutls/<data_name>/<model>_<cut_off>.csv`
- Existing result file policy: print `WARN` and skip

```bash
# single
bash run_ml_single.sh Acinetobacter 32 rf

# batch
bash run_ml_batch.sh
```

## Evaluations

We will add a dedicated results-analysis script to summarize model outputs under `resutls/`.

## Notebooks

- `notebooks/cutoff.ipynb`: visualize `datasets/data_pos_ratio.csv`
- `notebooks/results_plots.ipynb`: visualize and compare model outputs from `resutls/*/*.csv`

## Citations
