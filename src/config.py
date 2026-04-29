# PatentLens — Central Configuration
# All settings live here. Never hardcode these in other files.

# Model settings
PROVIDER = "novita"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = 0.1
MAX_TOKENS = 15

# The 8 technology domain categories
CATEGORIES = [
    "CLEAN_ENERGY",
    "AGRITECH",
    "MEDTECH",
    "MANUFACTURING",
    "DIGITAL_INFRA",
    "ENVIROTECH",
    "TRANSPORT",
    "CONSTRUCTION"
]

# Category definitions (used in prompts)
CATEGORY_DEFINITIONS = {
    "CLEAN_ENERGY":   "Solar, wind, hydrogen, battery, energy harvesting",
    "AGRITECH":       "Farming, crop monitoring, irrigation, soil analysis",
    "MEDTECH":        "Medical devices, diagnostics, drug delivery, health monitors",
    "MANUFACTURING":  "3D printing, robotics, new materials, automation",
    "DIGITAL_INFRA":  "IoT, wireless communication, embedded systems, edge computing",
    "ENVIROTECH":     "Water purification, waste management, pollution monitoring",
    "TRANSPORT":      "Electric vehicles, autonomous systems, traffic management",
    "CONSTRUCTION":   "Smart buildings, structural monitoring, construction materials"
}

# Data settings
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
MIN_ABSTRACT_WORDS = 50
SAMPLES_PER_CATEGORY = 700