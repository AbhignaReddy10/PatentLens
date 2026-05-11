import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from tqdm import tqdm
from config import CATEGORIES, CATEGORY_DEFINITIONS

# ── Search keywords for each category ────────────────────────────────────────
# Each category has multiple search queries
# We run each query and collect abstracts until we hit our target

SEARCH_QUERIES = {
    "CLEAN_ENERGY": [
        "solar cell semiconductor photovoltaic SME",
        "wind turbine blade energy harvesting startup",
        "hydrogen fuel cell membrane electrode",
        "lithium battery cathode anode energy storage",
        "thermoelectric energy harvesting generator",
        "perovskite solar panel efficiency",
        "biomass gasification renewable energy"
    ],
    "AGRITECH": [
        "precision agriculture soil moisture sensor",
        "crop monitoring drone multispectral imaging",
        "smart irrigation water management system",
        "vertical farming LED lighting hydroponics",
        "pest detection machine learning agriculture",
        "food processing packaging preservation",
        "greenhouse climate control automation"
    ],
    "MEDTECH": [
        "wearable biosensor glucose monitoring patch",
        "drug delivery nanoparticle targeted therapy",
        "diagnostic device point of care testing",
        "surgical instrument minimally invasive tool",
        "prosthetic limb neural interface device",
        "medical imaging portable ultrasound",
        "wound healing antimicrobial dressing"
    ],
    "MANUFACTURING": [
        "3D printing additive manufacturing composite",
        "industrial robot automation assembly line",
        "nanomaterial carbon nanotube fabrication",
        "CNC machining precision manufacturing tool",
        "quality control inspection vision system",
        "metal alloy heat treatment process",
        "polymer extrusion injection molding"
    ],
    "DIGITAL_INFRA": [
        "IoT sensor network wireless communication",
        "LoRa LoRaWAN edge computing embedded system",
        "5G antenna millimeter wave communication",
        "RFID tracking supply chain management",
        "smart meter energy monitoring grid",
        "network protocol low power wide area",
        "Bluetooth mesh sensor data transmission"
    ],
    "ENVIROTECH": [
        "water purification membrane filtration system",
        "wastewater treatment biological reactor",
        "air quality monitoring pollution sensor",
        "plastic recycling chemical process",
        "soil remediation contamination cleanup",
        "carbon capture sequestration method",
        "solid waste management composting system"
    ],
    "TRANSPORT": [
        "electric vehicle battery management system",
        "autonomous vehicle LiDAR navigation sensor",
        "EV charging station wireless inductive",
        "traffic management intelligent transportation",
        "drone delivery unmanned aerial vehicle",
        "hydrogen fuel cell vehicle powertrain",
        "bicycle scooter micro mobility sharing"
    ],
    "CONSTRUCTION": [
        "smart building energy management system",
        "structural health monitoring sensor bridge",
        "concrete strength durability construction",
        "insulation thermal acoustic building material",
        "prefabricated modular construction system",
        "facade cladding weatherproof building",
        "fire detection suppression safety system"
    ]
}

TARGET_PER_CATEGORY = 700  # we want 700 abstracts per category
MIN_ABSTRACT_LENGTH = 50   # minimum words in an abstract

def search_patents(query, max_results=100):
    """
    Search Google Patents and return list of abstracts.
    """
    abstracts = []

    # Google Patents search URL
    base_url = "https://patents.google.com/xhr/query"

    params = {
        "url": f"q={query.replace(' ', '+')}&language=ENGLISH&type=PATENT",
        "exp": "",
        "tags": ""
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract patent results
            results = data.get("results", {}).get("cluster", [])

            for cluster in results:
                patents = cluster.get("result", [])
                for patent in patents:
                    patent_data = patent.get("patent", {})
                    abstract = patent_data.get("abstract", "")

                    # Clean the abstract
                    abstract = abstract.strip()

                    # Only keep abstracts with enough words
                    if abstract and len(abstract.split()) >= MIN_ABSTRACT_LENGTH:
                        abstracts.append(abstract)

                    if len(abstracts) >= max_results:
                        break

    except Exception as e:
        print(f"  Error: {e}")

    return abstracts


def collect_category(category, queries, target=700):
    """
    Collect abstracts for one category using multiple search queries.
    """
    all_abstracts = []
    seen = set()  # to avoid duplicates

    print(f"\nCollecting: {category}")
    print(f"Target: {target} abstracts")

    for query in queries:
        if len(all_abstracts) >= target:
            break

        print(f"  Searching: '{query}'")
        abstracts = search_patents(query, max_results=150)

        for abstract in abstracts:
            # Skip duplicates
            abstract_key = abstract[:100]
            if abstract_key not in seen:
                seen.add(abstract_key)
                all_abstracts.append(abstract)

        print(f"  Collected so far: {len(all_abstracts)}")
        time.sleep(2)  # be respectful to Google's servers

    # Trim to target
    all_abstracts = all_abstracts[:target]

    print(f"  Final count for {category}: {len(all_abstracts)}")
    return all_abstracts


def main():
    print("=" * 55)
    print("PatentLens — InduSTex-9 Data Collection")
    print("=" * 55)

    all_data = []

    for category in CATEGORIES:
        queries = SEARCH_QUERIES[category]
        abstracts = collect_category(category, queries, TARGET_PER_CATEGORY)

        # Add to dataset with label
        for abstract in abstracts:
            all_data.append({
                "abstract": abstract,
                "label": category
            })

        # Save after each category in case something crashes
        df_temp = pd.DataFrame(all_data)
        df_temp.to_csv("data/instustex9_partial.csv", index=False)
        print(f"  Saved checkpoint: {len(df_temp)} total abstracts so far")

    # Final dataset
    df = pd.DataFrame(all_data)

    print("\n" + "=" * 55)
    print(f"Collection complete!")
    print(f"Total abstracts: {len(df)}")
    print(f"\nDistribution:")
    print(df["label"].value_counts())

    # Save final CSV
    df.to_csv("data/instustex9_raw.csv", index=False)
    print(f"\nSaved to: data/instustex9_raw.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()