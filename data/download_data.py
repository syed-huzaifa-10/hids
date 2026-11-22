"""Download the two NSL-KDD files used in this project into data/.

    python data/download_data.py
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
for name in ["KDDTrain+.txt", "KDDTest+.txt"]:
    dest = Path(__file__).parent / name
    print("downloading", name, "->", dest)
    urllib.request.urlretrieve(f"{BASE}/{name}", dest)
print("done")
