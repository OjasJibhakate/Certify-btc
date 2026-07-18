"""
train.py  —  main training entry point (implemented in Phase 3).

For now this only prints the active configuration so you can confirm the
local/cloud switch works before any model code exists.

    python train.py
"""

import config


def main():
    config.summary()
    print("\n[Phase 0] Scaffold only — no training loop yet.")
    print("Next: Phase 1 (datasets.py + preprocessing.py).")


if __name__ == "__main__":
    main()
