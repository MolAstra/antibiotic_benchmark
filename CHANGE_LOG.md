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
