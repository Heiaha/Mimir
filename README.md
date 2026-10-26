# Mimir
Mimir is a suite of scripts used for training a NNUE evaluation function for [Weiawaga](https://github.com/Heiaha/Weiawaga). 

# Setup 
To use it, some setup is required, as the position vector generation is done via PyO3 bindings to a Rust library. You will need the [Rust toolchain](https://www.rust-lang.org/tools/install) and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. Once you have these, create the environment (this also builds fen_parser, and rebuilds it whenever its Rust source changes) via

```
uv sync
```

and run scripts with `uv run`, e.g. `uv run train.py`.
