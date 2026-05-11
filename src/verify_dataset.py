import pandas as pd

print("=" * 50)
print("PatentLens — Dataset Verification")
print("=" * 50)

# Load all splits
full  = pd.read_csv("data/instustex9_full.csv")
train = pd.read_csv("data/train.csv")
val   = pd.read_csv("data/val.csv")
test  = pd.read_csv("data/test.csv")

# ── Basic counts ──────────────────────────────────────────
print(f"\nFull dataset : {len(full):,} samples")
print(f"Train split  : {len(train):,} samples")
print(f"Val split    : {len(val):,} samples")
print(f"Test split   : {len(test):,} samples")
print(f"Total splits : {len(train)+len(val)+len(test):,} samples")

# ── Label distribution ────────────────────────────────────
print(f"\nLabel distribution (full dataset):")
print(full["label"].value_counts())

# ── Check for problems ────────────────────────────────────
print(f"\nData quality checks:")

# Missing values
missing = full.isnull().sum()
print(f"  Missing values: {missing.sum()}")

# Duplicates
dupes = full.duplicated(subset=["abstract"]).sum()
print(f"  Duplicate abstracts: {dupes}")

# Abstract length
full["word_count"] = full["abstract"].str.split().str.len()
print(f"  Avg abstract length: {full['word_count'].mean():.0f} words")
print(f"  Min abstract length: {full['word_count'].min()} words")
print(f"  Max abstract length: {full['word_count'].max()} words")

# ── Show one sample per category ─────────────────────────
print(f"\nOne sample per category:")
for label in sorted(full["label"].unique()):
    sample = full[full["label"] == label]["abstract"].iloc[0]
    print(f"\n  [{label}]")
    print(f"  {sample[:120]}...")

# ── Check no data leakage between splits ─────────────────
print(f"\nLeakage check:")
train_texts = set(train["abstract"].tolist())
val_texts   = set(val["abstract"].tolist())
test_texts  = set(test["abstract"].tolist())

train_val_overlap  = len(train_texts & val_texts)
train_test_overlap = len(train_texts & test_texts)
val_test_overlap   = len(val_texts & test_texts)

print(f"  Train/Val overlap  : {train_val_overlap} (should be 0)")
print(f"  Train/Test overlap : {train_test_overlap} (should be 0)")
print(f"  Val/Test overlap   : {val_test_overlap} (should be 0)")

print("\n" + "=" * 50)
if missing.sum() == 0 and dupes == 0 and train_val_overlap == 0:
    print("Dataset is CLEAN and ready for experiments.")
    print("Phase 2 is COMPLETE.")
else:
    print("Issues found — check above and fix before Phase 3.")
print("=" * 50)