"""Safe weight inspection: lists array names/shapes of .npz (allow_pickle=False) and members of .zip.
Refuses pickle-based files (.pt/.pth/.pkl/.joblib/.ckpt)."""
import sys, zipfile
import numpy as np
REFUSE = (".pt", ".pth", ".pkl", ".joblib", ".ckpt")
def main(path):
    if path.lower().endswith(REFUSE):
        sys.exit(f"refusing pickle-based file: {path} (load only inside a sandbox)")
    if path.lower().endswith(".zip"):
        for i in zipfile.ZipFile(path).infolist(): print(f"{i.file_size:>12}  {i.filename}")
        return
    z = np.load(path, allow_pickle=False)
    total = 0
    for k in z.files:
        s = tuple(z[k].shape); n = int(np.prod(s)) if s else 1; total += n
        print(f"{k:40s} {str(s):20s} {z[k].dtype}")
    print("total elements:", f"{total:,}")
if __name__ == "__main__":
    if len(sys.argv) != 2: sys.exit("usage: inspect_npz.py FILE")
    main(sys.argv[1])
