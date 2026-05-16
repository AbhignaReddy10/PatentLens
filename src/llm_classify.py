# ─────────────────────────────────────────────────────────────
# PatentLens — Phase 3: LLM Classification Experiments
# Runs zero-shot and few-shot experiments on the test set
# using Llama-3.1-8B via HuggingFace (novita provider)
# ─────────────────────────────────────────────────────────────

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import pandas as pd
import time
import os
import sys

load_dotenv()

# ── Configuration ─────────────────────────────────────────────
PROVIDER = "novita"
MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

CATEGORIES = [
    "CLEAN_ENERGY",
    "AGRITECH",
    "MEDTECH",
    "MANUFACTURING",
    "DIGITAL_INFRA",
    "ENVIROTECH"
]

CATEGORY_DEFINITIONS = {
    "CLEAN_ENERGY":  "Solar, wind, hydrogen, battery, energy harvesting",
    "AGRITECH":      "Farming, crop monitoring, irrigation, soil analysis",
    "MEDTECH":       "Medical devices, diagnostics, drug delivery, health monitors",
    "MANUFACTURING": "3D printing, robotics, new materials, automation",
    "DIGITAL_INFRA": "IoT, wireless communication, embedded systems, edge computing",
    "ENVIROTECH":    "Water purification, waste management, pollution monitoring"
}

DELAY_BETWEEN_CALLS = 1.5  # seconds — avoids rate limiting

# ── Connect to API ────────────────────────────────────────────
print("Connecting to Llama-3.1-8B via HuggingFace...")
client = InferenceClient(provider=PROVIDER, api_key=HF_TOKEN)
print("Connected.\n")

# ── Build category list string for prompts ────────────────────
CATEGORY_LIST = "\n".join([
    f"{cat} - {CATEGORY_DEFINITIONS[cat]}"
    for cat in CATEGORIES
])

# ── Few-shot examples (3 per category = 18 total) ─────────────
# These are hand-picked clear examples from our dataset
FEW_SHOT_EXAMPLES = [
    ("CLEAN_ENERGY",
     "This study examines the efficiency of perovskite solar cells under varying irradiance conditions, demonstrating improved energy conversion through nanostructured interface layers."),
    ("CLEAN_ENERGY",
     "A novel wind turbine blade design using composite materials reduces aerodynamic drag and improves power output at low wind speeds."),
    ("CLEAN_ENERGY",
     "Lithium-sulfur battery technology with improved cathode materials achieves higher energy density for grid-scale storage applications."),

    ("AGRITECH",
     "Precision irrigation systems using wireless soil moisture sensors reduce water consumption by 40% while maintaining crop yield in semi-arid regions."),
    ("AGRITECH",
     "Machine learning algorithms applied to satellite imagery enable early detection of crop disease across large agricultural areas."),
    ("AGRITECH",
     "Smallholder farmers in developing regions benefit from mobile-based advisory systems that provide real-time crop management recommendations."),

    ("MEDTECH",
     "A wearable electrochemical biosensor continuously monitors glucose levels through sweat analysis using flexible printed electronics."),
    ("MEDTECH",
     "Point-of-care diagnostic devices using microfluidic technology enable rapid detection of infectious diseases in resource-limited settings."),
    ("MEDTECH",
     "Targeted nanoparticle drug delivery systems improve chemotherapy efficacy by concentrating therapeutic agents at tumor sites."),

    ("MANUFACTURING",
     "Fused deposition modelling with carbon fibre reinforced polymers produces high-strength components for aerospace applications."),
    ("MANUFACTURING",
     "Industrial robots equipped with computer vision systems perform quality inspection tasks with greater accuracy than human operators."),
    ("MANUFACTURING",
     "Industry 4.0 implementation in small manufacturing firms improves production efficiency through real-time process monitoring."),

    ("DIGITAL_INFRA",
     "LoRaWAN-based IoT networks enable low-power environmental monitoring across large urban areas with minimal infrastructure."),
    ("DIGITAL_INFRA",
     "Edge computing architectures reduce latency in industrial IoT applications by processing sensor data locally rather than in the cloud."),
    ("DIGITAL_INFRA",
     "5G network deployment in rural areas bridges the digital divide by providing high-speed connectivity to underserved communities."),

    ("ENVIROTECH",
     "Membrane bioreactor technology improves wastewater treatment efficiency while reducing energy consumption and sludge production."),
    ("ENVIROTECH",
     "Low-cost air quality monitoring networks using electrochemical sensors provide real-time pollution data for urban planning."),
    ("ENVIROTECH",
     "Photocatalytic water purification systems using titanium dioxide remove pharmaceutical contaminants from drinking water sources.")
]


def build_zeroshot_prompt(abstract):
    """
    Zero-shot prompt — no examples given.
    Just the category definitions and the abstract.
    """
    return f"""You are an expert in technology classification for UN SDG #9 monitoring.

Classify the following text into exactly one of these 6 categories:

{CATEGORY_LIST}

Text: {abstract}

Reply with ONLY the category label. No explanation. No punctuation.
Example reply: CLEAN_ENERGY"""


def build_fewshot_prompt(abstract):
    """
    Few-shot prompt — 18 labeled examples given (3 per category).
    Examples help the model understand the classification boundaries.
    """
    examples_text = "\n".join([
        f'Text: "{ex[1][:120]}..."\nLabel: {ex[0]}\n'
        for ex in FEW_SHOT_EXAMPLES
    ])

    return f"""You are an expert in technology classification for UN SDG #9 monitoring.

Classify texts into exactly one of these 6 categories:

{CATEGORY_LIST}

Here are some examples of correct classifications:

{examples_text}

Now classify this text:
Text: {abstract}

Reply with ONLY the category label. No explanation. No punctuation.
Example reply: CLEAN_ENERGY"""


def call_llm(prompt, retries=3):
    """
    Call the LLM API with retry logic.
    Returns the predicted label as a string.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait = 60 * (attempt + 1)
                print(f"  Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error (attempt {attempt+1}): {error_msg[:80]}")
                time.sleep(5)

    return "ERROR"


def clean_label(raw_label):
    """
    Clean the model's output to extract a valid category label.
    Models sometimes return extra text — this extracts just the label.
    """
    raw_upper = raw_label.upper().strip()

    # Direct match
    for cat in CATEGORIES:
        if cat in raw_upper:
            return cat

    # Partial match fallback
    for cat in CATEGORIES:
        cat_words = cat.replace("_", " ")
        if cat_words in raw_upper:
            return cat

    # If nothing matches return UNKNOWN
    return "UNKNOWN"


def run_experiment(test_df, experiment_name, prompt_fn):
    """
    Run one full experiment (zero-shot or few-shot).

    Args:
        test_df       : DataFrame with 'abstract' and 'label' columns
        experiment_name: 'zeroshot' or 'fewshot'
        prompt_fn     : function that builds the prompt

    Returns:
        DataFrame with predictions added
    """
    print(f"\n{'='*55}")
    print(f"Running experiment: {experiment_name.upper()}")
    print(f"Test samples: {len(test_df)}")
    print(f"Model: {MODEL}")
    print(f"{'='*55}\n")

    results = []
    correct = 0
    total = 0

    for i, row in test_df.iterrows():
        abstract = row["abstract"]
        true_label = row["label"]
        total += 1

        # Build prompt
        prompt = prompt_fn(abstract)

        # Call API
        raw_response = call_llm(prompt)
        predicted_label = clean_label(raw_response)

        # Track accuracy
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1

        results.append({
            "abstract":        abstract,
            "true_label":      true_label,
            "predicted_label": predicted_label,
            "raw_response":    raw_response,
            "correct":         is_correct
        })

        # Progress update every 10 samples
        if total % 10 == 0:
            running_acc = correct / total * 100
            print(f"  Progress: {total}/{len(test_df)} | "
                  f"Running accuracy: {running_acc:.1f}%")

        # Save checkpoint every 50 samples
        if total % 50 == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_path = f"results/{experiment_name}_checkpoint.csv"
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Delay between calls
        time.sleep(DELAY_BETWEEN_CALLS)

    results_df = pd.DataFrame(results)

    # Save final predictions
    output_path = f"results/{experiment_name}_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved: {output_path}")

    final_acc = correct / total * 100
    print(f"Final accuracy: {final_acc:.1f}%")

    return results_df


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load test set
    print("Loading test set...")
    test_df = pd.read_csv("data/test.csv")
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution:")
    print(test_df["label"].value_counts())

    # Decide which experiment to run
    # Pass 'zeroshot' or 'fewshot' as command line argument
    # Default is zeroshot
    experiment = sys.argv[1] if len(sys.argv) > 1 else "zeroshot"

    if experiment == "zeroshot":
        results = run_experiment(test_df, "zeroshot", build_zeroshot_prompt)
    elif experiment == "fewshot":
        results = run_experiment(test_df, "fewshot", build_fewshot_prompt)
    else:
        print(f"Unknown experiment: {experiment}")
        print("Use: python src/llm_classify.py zeroshot")
        print("Or : python src/llm_classify.py fewshot")
        sys.exit(1)

    print(f"\nExperiment complete.")
    print(f"Run evaluate.py to see F1 scores:")
    print(f"  python src/evaluate.py {experiment}")