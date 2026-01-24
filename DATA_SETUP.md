# Data Setup Guide for NUCML-Next

This guide explains how to set up all required data files for NUCML-Next.

## Quick Summary

**Minimum Required:**
- X4Pro SQLite database (EXFOR experimental data)

**Recommended for Full Features:**
- AME2020/NUBASE2020 data files (5 files for tier-based features)

---

## Step 1: EXFOR Database (Required)

NUCML-Next uses IAEA EXFOR experimental nuclear cross-section data in X4Pro SQLite format.

### Option A: Sample Database (Testing)

A small sample database is included in the repository:

```bash
ls data/x4sqlite1_sample.db
# File is already present, ~40 MB
```

### Option B: Full Database (Production)

Download the complete EXFOR database for production use:

1. Visit: https://www-nds.iaea.org/x4/
2. Download: `x4sqlite1.db` (~2-4 GB)
3. Place in project root or `data/` directory

**Note:** The full database is NOT committed to GitHub due to size constraints.

---

## Step 2: AME2020/NUBASE2020 Data (Recommended)

NUCML-Next implements a **tier-based feature hierarchy** requiring multiple AME2020/NUBASE2020 files.

### Why Multiple Files?

Different nuclear properties come from different evaluation files:
- **mass_1.mas20.txt**: Basic nuclear masses and binding energies
- **rct1.mas20.txt**: Two-particle separation energies and decay Q-values
- **rct2_1.mas20.txt**: One-particle separation energies and reaction Q-values
- **nubase_4.mas20.txt**: Nuclear structure (spin, parity, half-life)
- **covariance.mas20.txt**: Uncertainty correlations (optional)

### Download All Files

```bash
# Navigate to data directory
cd data/

# Download AME2020 files (required for Tiers B, C, E)
wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct2_1.mas20.txt

# Download NUBASE2020 file (required for Tier D)
wget https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt

# Optional: Download covariance data
wget https://www-nds.iaea.org/amdc/ame2020/covariance.mas20.txt
```

### Verify Downloads

```bash
ls -lh data/*.mas20.txt
```

Expected output:
```
-rw-r--r-- 1 user user  24M covariance.mas20.txt  (optional)
-rw-r--r-- 1 user user 462K mass_1.mas20.txt
-rw-r--r-- 1 user user 500K rct1.mas20.txt
-rw-r--r-- 1 user user 499K rct2_1.mas20.txt
-rw-r--r-- 1 user user 5.8K nubase_4.mas20.txt
```

---

## Step 3: Ingest EXFOR Data

Convert X4Pro SQLite to Parquet format for efficient loading:

### Basic Ingestion (No AME2020)

```bash
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1_sample.db \
    --output data/exfor_processed.parquet
```

### With AME2020 Enrichment (Recommended)

```bash
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1_sample.db \
    --output data/exfor_processed.parquet \
    --ame2020 data/mass_1.mas20.txt
```

**Note:** During ingestion, only `mass_1.mas20.txt` is used for basic enrichment. All other AME2020/NUBASE2020 files are loaded on-demand during feature generation.

---

## Step 4: Verify Setup

Test that everything is working:

```python
from nucml_next.data import NucmlDataset
from nucml_next.data.enrichment import AME2020DataEnricher

# Test EXFOR data loading
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular'
)
print(f"✓ Loaded {len(dataset.df):,} EXFOR measurements")

# Test AME2020/NUBASE2020 enrichment
enricher = AME2020DataEnricher(data_dir='data/')
enricher.load_all()
print(f"✓ Available tiers: {enricher.get_available_tiers()}")

# Should show: ['A', 'B', 'C', 'D', 'E'] if all files present
```

---

## Tier-Based Features

With all AME2020/NUBASE2020 files in place, you can use the full tier system:

### Feature Counts by Tier

| Tier | Description | Features | Required Files |
|------|-------------|----------|----------------|
| **A** | Core | 14 | None (always available) |
| **B** | + Geometric | 16 | mass_1.mas20.txt |
| **C** | + Energetics | 23 | mass_1, rct1, rct2_1 |
| **D** | + Topological | 32 | nubase_4.mas20.txt |
| **E** | + Complete Q-values | 40 | rct1, rct2_1 |

### Example Usage

```python
from nucml_next.data import NucmlDataset
from nucml_next.data.selection import DataSelection

# Select features by tier
selection = DataSelection(
    projectile='neutron',
    energy_min=1e-5,
    energy_max=2e7,
    mt_mode='all_physical',
    tiers=['A', 'B', 'C', 'D']  # Use Tier D features
)

dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    selection=selection
)

# Generate tier-based features (automatic AME2020 loading)
df = dataset.to_tabular(mode='tier')
print(f"Generated {df.shape[1]} features including Tier D topological features")
```

---

## Troubleshooting

### Missing AME2020 Files

If AME2020/NUBASE2020 files are missing, you'll see warnings:

```
WARNING: nubase_4.mas20.txt not found - Tier D features unavailable
```

**Solution:** Download the missing file(s) from https://www-nds.iaea.org/amdc/

### Available Tiers Less Than Expected

```python
enricher.get_available_tiers()  # Returns ['A', 'B', 'C', 'E']
# Missing 'D' because nubase_4.mas20.txt not found
```

**Solution:** Check which files are present:
```bash
ls -lh data/*.mas20.txt
```

### Feature Generation Errors

If you request a tier that's unavailable:

```python
# Error if nubase_4.mas20.txt is missing
df = dataset.to_tabular(mode='tier', tiers=['D'])
```

**Solution:** Only request tiers that are available, or download the required files.

---

## File Format Details

### AME2020 Files

All AME2020 files use **fixed-width Fortran-style format**:
- Headers start with `1` (page break) or `0` (line feed)
- Data lines contain isotope information
- `#` indicates estimated (non-experimental) values
- `*` indicates non-calculable values

### NUBASE2020 File

NUBASE uses a different fixed-width format:
- Columns 1-3: Mass number (A)
- Columns 5-8: Atomic number + isomer state (ZZZi)
- Columns 89-102: Spin and parity (J^π)
- Columns 70-80: Half-life with units (ys, zs, as, ..., My, Gy, Ey)

---

## Citations

If you use AME2020 or NUBASE2020 data in your work, please cite:

**AME2020:**
```
W.J. Huang, M. Wang, F.G. Kondev, G. Audi, and S. Naimi,
"The AME 2020 atomic mass evaluation (I),"
Chinese Phys. C 45, 030002 (2021).

M. Wang, W.J. Huang, F.G. Kondev, G. Audi, and S. Naimi,
"The AME 2020 atomic mass evaluation (II),"
Chinese Phys. C 45, 030003 (2021).
```

**NUBASE2020:**
```
F.G. Kondev, M. Wang, W.J. Huang, S. Naimi, and G. Audi,
"The NUBASE2020 evaluation of nuclear physics properties,"
Chinese Phys. C 45, 030001 (2021).
```

---

## Complete Setup Checklist

- [ ] X4Pro SQLite database obtained (sample or full)
- [ ] `mass_1.mas20.txt` downloaded
- [ ] `rct1.mas20.txt` downloaded
- [ ] `rct2_1.mas20.txt` downloaded
- [ ] `nubase_4.mas20.txt` downloaded
- [ ] EXFOR data ingested to Parquet
- [ ] Verified tier system with `enricher.get_available_tiers()`
- [ ] Tested feature generation with `dataset.to_tabular(mode='tier')`

**Status: Ready for tier-based feature engineering and ML training!** ✓
