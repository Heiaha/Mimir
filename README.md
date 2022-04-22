# Mimir
Mimir is a suite of scripts used for training a NNUE evaluation function for [Weiawaga](https://github.com/Heiaha/Weiawaga). 

# Setup 
To use it, some setup is required, as the position vector generation is done via PyO3 bindings to a Rust library. Firstly you will need the [Rust toolchain](https://www.rust-lang.org/tools/install) installed. You'll also need the packages in ```requirements.txt``` installed via pip or conda. Once you have these, navigate to fen_parser and build via

```
maturin develop
```