import pandas as pd
from sklearn.model_selection import train_test_split

keep = ['CLEAN_ENERGY', 'AGRITECH', 'MEDTECH',
        'MANUFACTURING', 'DIGITAL_INFRA', 'ENVIROTECH']

# Load and filter to 6 categories only
full = pd.read_csv('data/instustex9_full.csv')
print(f"Loaded: {len(full)} samples")
print(f"Columns: {full.columns.tolist()}")

# Filter to keep only our 6 categories
full = full[full['label'].isin(keep)].copy()
print(f"After filtering: {len(full)} samples")
print(full['label'].value_counts())

# Balance to 299 per category
balanced = []
for category in keep:
    cat_df = full[full['label'] == category]
    n = min(len(cat_df), 299)
    sampled = cat_df.sample(n=n, random_state=42)
    balanced.append(sampled)
    print(f"  {category}: {n} samples")

full = pd.concat(balanced, ignore_index=True)

# Shuffle
full = full.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nBalanced total: {len(full)}")

# Split
train_val, test = train_test_split(
    full, test_size=0.15, random_state=42, stratify=full['label']
)
train, val = train_test_split(
    train_val, test_size=0.176, random_state=42, stratify=train_val['label']
)

# Save
full.to_csv('data/instustex9_full.csv', index=False)
train.to_csv('data/train.csv', index=False)
val.to_csv('data/val.csv', index=False)
test.to_csv('data/test.csv', index=False)

print(f"\nDone!")
print(f"Full  : {len(full)}")
print(f"Train : {len(train)}")
print(f"Val   : {len(val)}")
print(f"Test  : {len(test)}")
print(f"\nFinal distribution:")
print(full['label'].value_counts())