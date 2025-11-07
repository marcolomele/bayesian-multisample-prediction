import os
import pandas as pd
from tqdm import tqdm

data_dir = os.path.join("data", "namesbystate")
rows = []

txt_files = [f for f in os.listdir(data_dir) if f.endswith(".TXT")]

for filename in tqdm(txt_files, desc="Processing state files"):
    filepath = os.path.join(data_dir, filename)
    state = filename.split('.')[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:
                # Format: state,sex,year,name,count
                rows.append({
                    "state": parts[0],
                    "sex": parts[1],
                    "year": int(parts[2]),
                    "name": parts[3],
                    "count": int(parts[4]),
                })

df = pd.DataFrame(rows, columns=["state", "sex", "year", "name", "count"])
output_path = os.path.join("data", "namesbystate.csv")
df.to_csv(output_path, index=False)
