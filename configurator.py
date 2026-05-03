"""
Minimal configurator compatible with nanoGPT train.py.
Usage examples:
  python train.py config/finetune.py --batch_size=8 --compile=False
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        # treat as a python config file path
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]
        if key not in globals():
            raise ValueError(f"Unknown config key: {key}")
        try:
            attempt = literal_eval(val)
        except Exception:
            attempt = val
        assert type(attempt) == type(globals()[key]), (
            f"Type mismatch for {key}: expected {type(globals()[key])}, got {type(attempt)}"
        )
        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
