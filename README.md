# NUCML-Next: Next-Generation Nuclear Data Evaluation

**Physics-Informed Deep Learning for Nuclear Cross-Section Prediction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

NUCML-Next is an **educational platform** for teaching physics-informed machine learning in nuclear engineering. It demonstrates the evolution from classical ML (XGBoost, Decision Trees) to physics-informed deep learning (GNNs + Transformers) for nuclear data evaluation.

### The Problem We Solve

**The Validation Paradox:**
> Low MSE on test data ≠ Safe reactor predictions!

Classical ML models can achieve low error on geometric metrics while producing unphysical and unsafe reactor predictions.

**The Solution:**
- **Graph Neural Networks** → Learn nuclear topology
- **Transformers** → Smooth cross-section curves
- **Physics-Informed Loss** → Enforce constraints
- **Sensitivity Weighting** → Prioritize reactor-critical reactions

---

## Quick Start

### 📚 Educational Mode (Synthetic Data for Teaching)
```bash
# Install
git clone https://github.com/WHopkins-git/NUCML-Next.git
cd NUCML-Next
pip install -r requirements.txt

# Run educational notebooks (auto-generates synthetic demo data)
jupyter notebook notebooks/00_Baselines_and_Limitations.ipynb
```

### 🏭 Production Mode (Real EXFOR Data Only)
```bash
# 1. Download EXFOR-X5json from IAEA
#    Visit: https://www-nds.iaea.org/exfor/
#    Download and unzip bulk database

# 2. Ingest EXFOR to Parquet
python scripts/ingest_exfor.py \
    --exfor-root ~/data/EXFOR-X5json/ \
    --output data/exfor_processed.parquet

# 3. Verify real data loading
jupyter notebook notebooks/00_Production_EXFOR_Data_Loading.ipynb
```

**⚠️ PRODUCTION REQUIREMENT:**
For production evaluations, you MUST use real EXFOR data:
```python
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    require_real_data=True  # ← Enforces real data, prevents synthetic fallback
)
dataset.assert_real_data()  # ← Verification check
```

---

## Data Setup (Production Mode with EXFOR)

NUCML-Next v1.1.0-alpha now supports **real-world nuclear cross-section data** from the IAEA EXFOR database.

### Option 1: Download EXFOR-X5json Bulk Database

1. Download the EXFOR-X5json bulk database from IAEA:
   - Visit: https://www-nds.iaea.org/exfor/
   - Download: EXFOR-X5json bulk ZIP file (~500 MB compressed)
   - Unzip to a local directory, e.g., `~/data/EXFOR-X5json/`

2. Run the EXFOR ingestor:

```python
from nucml_next.data import ingest_exfor

# Ingest EXFOR database to Parquet
df = ingest_exfor(
    exfor_root='~/data/EXFOR-X5json/',
    output_path='data/exfor_processed.parquet',
    ame2020_path=None,  # Will use SEMF approximation
    max_files=None      # Process all files (set to 100 for testing)
)
```

3. Use the processed data:

```python
from nucml_next.data import NucmlDataset

# Load processed EXFOR data
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='graph',
    filters={'Z': [92], 'MT': [18, 102]}  # U reactions only
)
```

### Option 2: Quick Start with Synthetic Data

For educational purposes or quick testing, NUCML-Next generates synthetic data automatically:

```python
from nucml_next.data import NucmlDataset

# Generates synthetic U-235, U-238, Pu-239 data
dataset = NucmlDataset(data_path=None, mode='graph')
```

### AME2020 Integration (Optional)

For enhanced isotope features, download AME2020 data:

```bash
wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt -O data/ame2020.txt
```

Then use it during ingestion:

```python
df = ingest_exfor(
    exfor_root='~/data/EXFOR-X5json/',
    output_path='data/exfor_processed.parquet',
    ame2020_path='data/ame2020.txt'  # Real AME2020 data
)
```

---

## Features

### v1.1.0-alpha (Production-Ready)
✓ **EXFOR-X5json bulk ingestor** with AME2020 enrichment
✓ **Partitioned Parquet** data fabric for large-scale datasets
✓ **Real experimental data** from IAEA EXFOR database

### Core Framework
✓ **Dual-view data architecture** (Graph + Tabular)
✓ **Baseline models** (Decision Trees, XGBoost)
✓ **GNN-Transformer** architecture
✓ **Physics-informed loss** functions
✓ **OpenMC integration** for validation
✓ **Sensitivity analysis** for reactor safety

---

## Documentation

See the [Wiki](https://github.com/WHopkins-git/NUCML-Next/wiki) for full documentation and tutorials.

---

## License

MIT License - See LICENSE file for details.