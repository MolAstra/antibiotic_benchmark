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

```bash
# open notebook
jupyter lab 0_data_preparation.ipynb

# optional: run notebook headless
# jupyter nbconvert --to notebook --execute 0_data_preparation.ipynb --output 0_data_preparation.executed.ipynb
```

`0_data_preparation.ipynb` prepares and standardizes the input table, then exports genus-level datasets used by downstream DPNet split and modeling scripts.

## Data Spliting

## Modeling

## Evaluations

## Notebooks

## Citations
