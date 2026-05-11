# PatentLens 🔬

> Benchmarking Zero-Shot, Few-Shot, and Fine-Tuned Large Language Models  
> for SME Innovation Patent Classification in Support of UN SDG #9

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Llama3.1-orange.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SDG](https://img.shields.io/badge/UN%20SDG-%239%20Industry%20%26%20Innovation-red.svg)](https://sdgs.un.org/goals/goal9)

---

## The Problem

Over **3.4 million patents** are filed globally every year by startups and SMEs.  
Organizations tracking innovation for UN SDG #9 monitoring cannot read them manually.  
PatentLens automates technology domain classification of patent abstracts using LLMs.

## The Research Question

> When classifying SME innovation patents by technology domain,  
> do fine-tuned transformer models outperform zero-shot LLM APIs —  
> and is the performance gap worth the computational cost?

---

## What We Build

| Component | Description |
|---|---|
| **InduSTex-9 Dataset** | 5,000 labeled SME patent abstracts across 8 SDG #9 domains |
| **LLM Benchmark** | Zero-shot and few-shot evaluation of Llama-3.1-8B |
| **Fine-tuned Models** | BERT-base and RoBERTa-base trained on patent data |
| **FastAPI Backend** | REST API serving real-time predictions |
| **Live Demo** | Gradio app deployed on HuggingFace Spaces |
| **Research Paper** | Published on arXiv |

---

## 8 Technology Domain Categories

| Label | Domain | Description |
|---|---|---|
| `CLEAN_ENERGY` | Clean Energy | Solar, wind, hydrogen, battery, energy harvesting |
| `AGRITECH` | Agricultural Technology | Farming, crop monitoring, irrigation, soil analysis |
| `MEDTECH` | Medical Technology | Devices, diagnostics, drug delivery, health monitors |
| `MANUFACTURING` | Advanced Manufacturing | 3D printing, robotics, new materials, automation |
| `DIGITAL_INFRA` | Digital Infrastructure | IoT, wireless communication, embedded systems |
| `ENVIROTECH` | Environmental Technology | Water purification, waste management, pollution |
| `TRANSPORT` | Transportation | Electric vehicles, autonomous systems, traffic |
| `CONSTRUCTION` | Smart Construction | Buildings, structural monitoring, civil engineering |

---

## Tech Stack
_ for the tech stack we will be using foru ai maodels which are currently being taken form the hugging face 