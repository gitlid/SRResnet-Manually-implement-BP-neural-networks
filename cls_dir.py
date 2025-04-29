from pathlib import Path
import shutil

check = Path("checkpoints")

for chc in check.glob("*"):
    if not any((chc/"imgs").iterdir()) or not any((chc/"params").iterdir()):
        shutil.rmtree(chc)
        print(f"{chc} deleted")
