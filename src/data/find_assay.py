from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

file = PROJECT_ROOT / "data" / "raw" / "DMS_substitutions.csv"

df = pd.read_csv(file)