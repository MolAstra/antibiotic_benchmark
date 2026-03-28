# CHANGE_LOG

## 2026-03-28

- Added `dpnet` installation command in README setup section:
  - `pip install https://github.com/zhaisilong/dpnet.git`
- Added submodule installation and update tutorial in README:
  - Clone with `--recurse-submodules`
  - Initialize existing clone submodules
  - Sync/update submodules after pulling latest repo changes
- Consolidated setup into a one-pass install flow under `## Setups`:
  - Clone + submodule bootstrap + environment + dependencies + editable install
  - Kept submodule update commands as optional inline comments in the same setup block
- Fixed `dpnet` install command format to pip VCS syntax:
  - from `pip install https://github.com/zhaisilong/dpnet.git`
  - to `pip install "git+https://github.com/zhaisilong/dpnet.git"`
- Added `1_cutoff_analysis.py` for MIC cutoff sliding analysis on `datasets/genus_top15`:
  - Two-fold dilution-grid cutoff scan (plus zero)
  - Per-genus RF + StratifiedKFold evaluation
  - Dual label directions (`resistant_pos` and `sensitive_pos`)
  - CSV outputs: full scan results and best cutoff summary
- Updated cutoff strategy to fixed MIC levels for comparability:
  - Default fixed levels: `2,4,8,16,32,64,128`
  - Added CLI option `--cutoff-levels` for explicit override
- Added `run_dpnet.py` to build DPNet task input from `datasets/genus_top15` and run processing:
  - Prepares `datasets/datasets/Acinetobacter/raw/Acinetobacter.csv`
  - Binarizes MIC `value` by cutoff (default `32`, resistant-positive)
  - Validates `task_meta.json` and calls `dpnet process ... --root_dir ...`
