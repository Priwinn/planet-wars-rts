"""Fix checkpoint `args` references from `config.ppo_config` -> `config_files.ppo_config`.

Usage:
    python -m util.fix_checkpoint_args /path/to/checkpoint.pt --inplace
    python -m util.fix_checkpoint_args /path/to/checkpoint.pt --out fixed.pt

The script will:
 - load the checkpoint via `torch.load`
 - look for the key `args` in the top-level state
 - recursively replace any string occurrences of the old module path with the new one
 - if `args` is an object whose class `__module__` equals the old path, update the class __module__ to the new path
 - save the modified checkpoint (in-place or to `--out`)

Note: This edits only Python-level string/module references inside the `args` structure. If your checkpoint embeds pickled class definitions from the old module path in other parts of the file, additional shims (a compatibility package) may be required. We already provide a `config` shim, but this script helps when the path is stored inside `args`.
"""

from __future__ import annotations
import argparse
import copy
import torch
from typing import Any

OLD = "config.ppo_config"
NEW = "config_files.ppo_config"


def replace_in_obj(obj: Any, old: str = OLD, new: str = NEW) -> Any:
    """Recursively replace occurrences of `old` with `new` inside Python objects.

    Handles: str, dict, list, tuple, set, and objects with __dict__.
    Returns a (possibly) new object.
    """
    # Strings
    if isinstance(obj, str):
        return obj.replace(old, new) if old in obj else obj

    # Dictionaries
    if isinstance(obj, dict):
        newd = {}
        for k, v in obj.items():
            newk = replace_in_obj(k, old, new) if isinstance(k, (str, dict, list, tuple, set)) else k
            newd[newk] = replace_in_obj(v, old, new)
        return newd

    # Lists
    if isinstance(obj, list):
        return [replace_in_obj(x, old, new) for x in obj]

    # Tuples
    if isinstance(obj, tuple):
        return tuple(replace_in_obj(list(obj), old, new))

    # Sets
    if isinstance(obj, set):
        return {replace_in_obj(x, old, new) for x in obj}

    # If object has __dict__, try to mutate attributes in-place
    if hasattr(obj, "__dict__"):
        for attr, val in vars(obj).items():
            try:
                replaced = replace_in_obj(val, old, new)
                setattr(obj, attr, replaced)
            except Exception:
                # If attribute is read-only or replacement fails, skip
                pass
        return obj

    # Fallback: return as-is
    return obj


def fix_checkpoint(path: str, out_path: str | None = None, inplace: bool = True) -> str:
    """Load checkpoint, fix `args`, and write back.

    Returns the path of the written file.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    modified = False

    if "args" not in state:
        print(f"Warning: 'args' key not found in checkpoint {path}; nothing to change.")
    else:
        args = state["args"]
        print(f"Loaded args type: {type(args)}")

        # If args is an object whose class module matches OLD, update the module attribute
        try:
            cls = args.__class__
            print(f"args.__class__.__module__ = {cls.__module__}")
            if getattr(cls, "__module__", None) == OLD:
                print(f"Updating args.__class__.__module__ from {OLD} -> {NEW}")
                cls.__module__ = NEW
                modified = True
        except Exception:
            pass

        # Recursively replace strings in args
        new_args = replace_in_obj(args, OLD, NEW)
        if new_args is not args:
            state["args"] = new_args
            modified = True

    if not modified:
        print("No modifications necessary.")


    # Determine output path
    if out_path:
        save_path = out_path
    elif inplace:
        save_path = path
    else:
        save_path = path + ".fixed.pt"

    torch.save(state, save_path)
    print(f"Saved fixed checkpoint to {save_path}")
    return save_path


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Fix checkpoint args module path from config.ppo_config to config_files.ppo_config")
    # parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    # parser.add_argument("--out", help="Output file path (if not provided, modifies in-place)")
    # parser.add_argument("--no-inplace", dest="inplace", action="store_false", help="Do not overwrite original file; write to <file>.fixed.pt")
    # args = parser.parse_args()
    # out = args.out if args.out else None
    # fix_checkpoint(args.checkpoint, out_path=out, inplace=args.inplace)

    # Recursive folder
    import os
    for root, dirs, files in os.walk("models/"):
        for file in files:
            if file.endswith(".pt"):
                path = os.path.join(root, file)
                fix_checkpoint(path, inplace=True)

