"""Inspect the Disco103 NPZ weights file to see all parameter names and shapes.

Usage (in Colab):
    !pip install git+https://github.com/google-deepmind/disco_rl.git
    Then run this script to see the weight structure.

    Or locally:
    python scripts/inspect_disco103.py path/to/disco_103.npz
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        # Try to find it via the installed package
        try:
            import disco_rl
            pkg_dir = Path(disco_rl.__file__).parent
            npz_path = pkg_dir / "update_rules" / "weights" / "disco_103.npz"
            if not npz_path.exists():
                print("disco_rl installed but disco_103.npz not found at expected path")
                print(f"  Looked in: {npz_path}")
                sys.exit(1)
        except ImportError:
            print("Usage: python scripts/inspect_disco103.py <path_to_disco_103.npz>")
            print()
            print("Or install disco_rl first:")
            print("  pip install git+https://github.com/google-deepmind/disco_rl.git")
            sys.exit(1)
    else:
        npz_path = sys.argv[1]

    import numpy as np
    data = np.load(npz_path, allow_pickle=True)

    print(f"\nDisco103 weights: {npz_path}")
    print(f"Total parameters: {len(data.files)}")
    print(f"{'Parameter':80s} {'Shape':>20s} {'Size':>10s}")
    print("-" * 115)

    total_size = 0
    for key in sorted(data.files):
        arr = data[key]
        size = arr.size
        total_size += size
        print(f"{key:80s} {str(arr.shape):>20s} {size:>10,d}")

    print("-" * 115)
    print(f"{'Total':80s} {'':>20s} {total_size:>10,d}")
    print(f"\nTotal trainable parameters: {total_size:,d}")


if __name__ == "__main__":
    main()
