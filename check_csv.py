import pandas as pd
import glob
import os

csv_files = glob.glob("D:/projectt/*.csv")
print(f"Found files in D:/projectt: {csv_files}")

for f in csv_files:
    print(f"\n========================================\nFile: {f}")
    try:
        # Read first 25 lines to inspect where the headers are
        with open(f, 'r') as file:
            print("First 25 lines:")
            for i in range(15):
                line = file.readline()
                if not line:
                    break
                print(f"{i+1}: {repr(line)}")
        
        # Read the file by skipping NetLogo metadata lines
        skiprows = 6
        with open(f, 'r') as file:
            for idx, line in enumerate(file):
                if "[run number]" in line or "attacker-skill" in line or "attacker_skill" in line or "attack-success" in line or "attack_success" in line:
                    skiprows = idx
                    break
        
        print(f"Detected header row at index: {skiprows}")
        df = pd.read_csv(f, skiprows=skiprows)
        print("Successfully loaded. Shape:", df.shape)
        print("Columns:", list(df.columns))
        
        # Look for target column
        target_cols = [c for c in df.columns if any(x in c.lower() for x in ["attack", "success", "risk", "target", "label"])]
        print("Potential target columns:", target_cols)
        for col in target_cols:
            try:
                # convert to numeric
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                print(f"Stats for {col}:")
                print(f"  Count: {len(series)}")
                print(f"  Mean:  {series.mean()}")
                print(f"  Min:   {series.min()}")
                print(f"  Max:   {series.max()}")
            except Exception as e:
                print(f"  Failed stats for {col}: {e}")
    except Exception as e:
        print("Error processing file:", e)
