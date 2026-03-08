import numpy as np
from pathlib import Path


def convert_npy_to_npz_recursive(in_path, out_path):
    in_path = Path(in_path)
    out_path = Path(out_path)

    npy_files = list(in_path.rglob("*.npy"))

    for npy_file in npy_files:
        # Relative path from input root
        rel_path = npy_file.relative_to(in_path)

        # Create output path with same folder structure
        out_file = (out_path / rel_path).with_suffix(".npz")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Load and compress
        data = np.load(npy_file, allow_pickle=True)
        np.savez_compressed(out_file, data)

        print(f"Converted: {npy_file} -> {out_file}")
        

if __name__ == "__main__":
    in_path = "/storage/ecd_separate"
    out_path = "/storage/ecd_separate_npz/"

    convert_npy_to_npz_recursive(in_path, out_path)

    