from datasets import load_dataset
import pandas as pd

print("Loading OSDG dataset from HuggingFace...")

dataset = load_dataset("albertmartinez/OSDG", split="train")
df = dataset.to_pandas()

print(f"Loaded successfully!")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# See what the label column contains
print(f"\nLabel distribution:")
print(df["label"].value_counts())

# Show first few rows
print(f"\nFirst 3 rows:")
print(df.head(3))