from datasets import load_dataset
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

print("=" * 55)
print("PatentLens — Building InduSTex-9 Dataset v2")
print("=" * 55)

# ── Step 1: Load ALL splits of OSDG ──────────────────────
print("\nStep 1: Loading OSDG dataset (all splits)...")

train_ds = load_dataset("albertmartinez/OSDG", split="train")
test_ds  = load_dataset("albertmartinez/OSDG", split="test")

df_train = train_ds.to_pandas()
df_test  = test_ds.to_pandas()
df       = pd.concat([df_train, df_test], ignore_index=True)

print(f"Total OSDG samples (train + test): {len(df)}")
print(f"Label distribution:")
print(df["label"].value_counts())

# ── Step 2: Use ALL SDGs — we classify by keyword anyway ─
print("\nStep 2: Using all SDG texts for keyword matching...")
# We use every text in the dataset — our keyword matching
# assigns the technology subcategory regardless of SDG label
all_texts = df.copy()
print(f"Total texts available: {len(all_texts)}")

# ── Step 3: Expanded keyword mapping ─────────────────────
print("\nStep 3: Assigning subcategory labels...")

KEYWORDS = {
    "CLEAN_ENERGY": [
        "solar", "photovoltaic", "wind energy", "wind turbine",
        "renewable energy", "hydrogen fuel", "battery storage",
        "energy storage", "biofuel", "geothermal", "hydropower",
        "clean energy", "green energy", "energy harvesting",
        "fuel cell", "biomass energy", "tidal energy",
        "solar panel", "wind power", "energy transition",
        "decarbonization", "low carbon energy", "net zero energy",
        "offshore wind", "solar farm", "energy efficiency"
    ],
    "AGRITECH": [
        "agriculture", "farming", "crop", "irrigation", "soil",
        "fertilizer", "pesticide", "livestock", "aquaculture",
        "precision farming", "food production", "harvest",
        "agricultural", "greenhouse", "seed", "agroforestry",
        "food security", "rural development", "smallholder",
        "farmer", "agri", "food system", "arable land",
        "crop yield", "plant disease", "animal husbandry",
        "fishery", "food supply chain", "agronomic"
    ],
    "MEDTECH": [
        "medical device", "healthcare technology", "diagnostic",
        "drug delivery", "wearable health", "biosensor",
        "telemedicine", "health innovation", "clinical device",
        "pharmaceutical", "vaccine technology", "health monitoring",
        "surgical device", "prosthetic", "medical imaging",
        "point of care", "mhealth", "digital health",
        "electronic health", "health tech", "biomedical",
        "medical equipment", "health system technology",
        "remote patient", "medical instrument", "ehealth"
    ],
    "MANUFACTURING": [
        "manufacturing", "3d printing", "additive manufacturing",
        "robotics", "automation", "industrial production",
        "factory", "supply chain", "quality control",
        "composite material", "nanotechnology", "smart factory",
        "industry 4.0", "production process", "machining",
        "industrial innovation", "lean manufacturing",
        "advanced manufacturing", "industrial technology",
        "production efficiency", "manufacturing process",
        "industrial robot", "computer aided manufacturing"
    ],
    "DIGITAL_INFRA": [
        "internet of things", "iot", "wireless network",
        "broadband", "digital infrastructure", "connectivity",
        "5g", "telecommunications", "smart city technology",
        "edge computing", "cloud computing", "digital platform",
        "information technology", "cybersecurity", "embedded system",
        "digital access", "internet access", "network infrastructure",
        "mobile network", "digital divide", "data center",
        "software platform", "digital service", "e-government"
    ],
    "ENVIROTECH": [
        "water treatment", "wastewater", "pollution control",
        "recycling technology", "waste management technology",
        "environmental monitoring", "clean water technology",
        "air quality technology", "carbon capture",
        "green technology", "circular economy technology",
        "environmental technology", "waste recycling",
        "pollution monitoring", "environmental sensor",
        "water purification", "sewage treatment",
        "hazardous waste", "emission reduction technology"
    ],
    "TRANSPORT": [
        "electric vehicle", "autonomous vehicle",
        "transportation technology", "mobility innovation",
        "logistics technology", "freight innovation",
        "aviation technology", "railway innovation",
        "shipping technology", "traffic management",
        "ev charging", "self-driving", "urban mobility",
        "public transport technology", "vehicle electrification",
        "transport infrastructure", "smart transport",
        "connected vehicle", "last mile delivery"
    ],
    "CONSTRUCTION": [
        "construction technology", "building technology",
        "infrastructure development", "bridge construction",
        "smart building", "structural engineering",
        "concrete innovation", "building material",
        "civil engineering innovation", "housing technology",
        "urban infrastructure", "road construction",
        "construction innovation", "building automation",
        "green building", "sustainable construction",
        "prefabricated building", "modular construction",
        "building information", "construction management"
    ]
}

def assign_category(text):
    text_lower = text.lower()
    scores = defaultdict(int)
    for category, keywords in KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[category] += 1
    if not scores:
        return None
    return max(scores, key=scores.get)

all_texts["subcategory"] = all_texts["text"].apply(assign_category)
labeled = all_texts.dropna(subset=["subcategory"]).copy()
print(f"Successfully labeled: {len(labeled)}")

# ── Step 4: Distribution ──────────────────────────────────
print("\nStep 4: Raw subcategory distribution:")
dist = labeled["subcategory"].value_counts()
print(dist)

# ── Step 5: Balance at 300 per category ──────────────────
print("\nStep 5: Balancing to 300 per category...")

TARGET = 300  # minimum credible sample size per category

balanced_dfs = []
for category in KEYWORDS.keys():
    cat_df = labeled[labeled["subcategory"] == category]
    available = len(cat_df)

    if available == 0:
        print(f"  WARNING: No samples found for {category}")
        continue

    n = min(TARGET, available)
    sampled = cat_df.sample(n=n, random_state=42)
    balanced_dfs.append(sampled)
    print(f"  {category}: {n} samples (available: {available})")

balanced = pd.concat(balanced_dfs, ignore_index=True)

# ── Step 6: Clean and format ──────────────────────────────
print("\nStep 6: Cleaning...")

final_df = balanced[["text", "subcategory"]].copy()
final_df.columns = ["abstract", "label"]
final_df = final_df.drop_duplicates(subset=["abstract"])
final_df = final_df[final_df["abstract"].str.split().str.len() >= 30]
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset size: {len(final_df)}")
print(f"\nFinal distribution:")
print(final_df["label"].value_counts())

# ── Step 7: Train / Val / Test split ─────────────────────
print("\nStep 7: Train/Val/Test split...")

train_val, test = train_test_split(
    final_df, test_size=0.15,
    random_state=42, stratify=final_df["label"]
)
train, val = train_test_split(
    train_val, test_size=0.176,
    random_state=42, stratify=train_val["label"]
)

print(f"Train : {len(train)}")
print(f"Val   : {len(val)}")
print(f"Test  : {len(test)}")

# ── Step 8: Save ──────────────────────────────────────────
print("\nStep 8: Saving...")

final_df.to_csv("data/instustex9_full.csv", index=False)
train.to_csv("data/train.csv",              index=False)
val.to_csv("data/val.csv",                  index=False)
test.to_csv("data/test.csv",                index=False)

print("\n" + "=" * 55)
print("Dataset build complete!")
print(f"Full  : data/instustex9_full.csv ({len(final_df)} samples)")
print(f"Train : data/train.csv           ({len(train)} samples)")
print(f"Val   : data/val.csv             ({len(val)} samples)")
print(f"Test  : data/test.csv            ({len(test)} samples)")
print("=" * 55)