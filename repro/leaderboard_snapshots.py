"""Reproduce the Simulation leaderboard snapshot table from SiamRahman29/pokemon-tcg-agent (out/lb*/*.zip)."""
import glob, io, os, sys, zipfile
import pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "SiamRahman29__pokemon-tcg-agent")
snaps = []
for z in sorted(glob.glob(os.path.join(root, "out", "lb*", "pokemon-tcg-ai-battle.zip"))):
    zf = zipfile.ZipFile(z)
    for n in zf.namelist():
        if n.endswith(".csv"):
            snaps.append((n.split("leaderboard-")[-1].replace(".csv", ""), pd.read_csv(io.BytesIO(zf.read(n)))))
snaps.sort(key=lambda x: x[0])
print("snapshot,teams,top,>=800,>=900,>=1000,>=1100,>=1200,median")
for ts, df in snaps:
    s = df["Score"].astype(float)
    print(",".join(map(str, [ts, len(df), round(s.max(), 1)] + [int((s >= t).sum()) for t in (800, 900, 1000, 1100, 1200)] + [round(s.median(), 1)])))
