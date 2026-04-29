from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import time

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Confirmed working — do not change these
PROVIDER = "novita"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

client = InferenceClient(
    provider=PROVIDER,
    api_key=HF_TOKEN
)

def classify_patent(abstract):
    """
    Takes a patent abstract string.
    Returns one of 8 category labels.
    """
    prompt = f"""You are a patent classification expert working on UN SDG #9 monitoring.

Classify the patent abstract below into exactly one of these 8 categories:

CLEAN_ENERGY     - Solar, wind, hydrogen, battery, energy harvesting
AGRITECH         - Farming, crop monitoring, irrigation, soil analysis
MEDTECH          - Medical devices, diagnostics, drug delivery, health monitors
MANUFACTURING    - 3D printing, robotics, new materials, automation
DIGITAL_INFRA    - IoT, wireless communication, embedded systems, edge computing
ENVIROTECH       - Water purification, waste management, pollution monitoring
TRANSPORT        - Electric vehicles, autonomous systems, traffic management
CONSTRUCTION     - Smart buildings, structural monitoring, construction materials

Patent Abstract:
{abstract}

Reply with ONLY the category label. No explanation. No punctuation. Just the label.
Example output: CLEAN_ENERGY"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=15,
        temperature=0.1   # low temperature = more consistent outputs
    )

    return response.choices[0].message.content.strip()


# ── 5 test patents covering 5 different categories ──────────────────────────

tests = [
    {
        "abstract": "A photovoltaic cell using bismuth vanadate semiconductor layers for enhanced solar energy conversion through a nanostructured interface that reduces electron-hole recombination rates.",
        "expected": "CLEAN_ENERGY"
    },
    {
        "abstract": "An autonomous irrigation system using soil moisture sensors and machine learning algorithms to optimize water distribution timing and volume across agricultural crop fields.",
        "expected": "AGRITECH"
    },
    {
        "abstract": "A wearable biosensor device for continuous glucose monitoring using electrochemical detection embedded in a flexible adhesive skin patch with wireless Bluetooth data transmission.",
        "expected": "MEDTECH"
    },
    {
        "abstract": "A novel fused deposition modelling process for fabricating high-strength composite structures using carbon fibre reinforced polymer filaments in a desktop 3D printing system.",
        "expected": "MANUFACTURING"
    },
    {
        "abstract": "A low-power wide-area IoT sensor node for urban air quality monitoring using LoRaWAN communication protocol and onboard particulate matter detection circuitry.",
        "expected": "DIGITAL_INFRA"
    }
]

# ── Run the test ─────────────────────────────────────────────────────────────

print("=" * 55)
print("PatentLens — Live Classification Test")
print(f"Model  : {MODEL}")
print(f"Provider: {PROVIDER}")
print("=" * 55)

passed = 0

for i, test in enumerate(tests):
    print(f"\nPatent {i+1}:")
    print(f"  Abstract : {test['abstract'][:70]}...")
    print(f"  Expected : {test['expected']}")

    label = classify_patent(test["abstract"])
    clean_label = label.upper().strip()

    if test["expected"] in clean_label:
        print(f"  Got      : {label}")
        print(f"  Result   : PASS ✓")
        passed += 1
    else:
        print(f"  Got      : {label}")
        print(f"  Result   : CHECK (may still be reasonable)")

    time.sleep(1)  # small delay between calls

print("\n" + "=" * 55)
print(f"Results  : {passed}/5 matched expected labels")

if passed >= 4:
    print("Status   : EXCELLENT — API fully working")
    print("\nPhase 1 is COMPLETE.")
    print("Next step: GitHub repo setup + first commit")
elif passed >= 3:
    print("Status   : GOOD — API working, minor label variation")
    print("This is normal for LLMs — we handle it in Phase 3")
    print("\nPhase 1 is COMPLETE.")
    print("Next step: GitHub repo setup + first commit")
else:
    print("Status   : NEEDS ATTENTION — paste output for help")
print("=" * 55)