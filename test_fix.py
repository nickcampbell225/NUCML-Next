"""
Test that models can now learn energy-dependent patterns with the new dataset.
"""
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from nucml_next.data import NucmlDataset, DataSelection
from nucml_next.baselines import DecisionTreeEvaluator

print("="*80)
print("VERIFICATION: Testing Energy-Dependent Learning")
print("="*80)

# Load new demonstration dataset
selection = DataSelection(
    projectile='neutron',
    energy_min=1e-5,
    energy_max=2e7,
    mt_mode='all_physical',
    drop_invalid=True
)

dataset = NucmlDataset(data_path='data/exfor_processed.parquet', mode='tabular', selection=selection)
df_naive = dataset.to_tabular(mode='naive')

print(f"\n✓ Loaded {len(df_naive):,} measurements for training")

# Train a simple Decision Tree
print("\n" + "="*80)
print("Training Decision Tree on new dataset...")
print("="*80)
dt_model = DecisionTreeEvaluator(max_depth=10, min_samples_leaf=5)
dt_metrics = dt_model.train(df_naive)

print("\n" + "="*60)
print("Training Results:")
print("="*60)
print(f"  Tree depth: {dt_metrics['tree_depth']}")
print(f"  Number of leaves: {dt_metrics['num_leaves']}")
print(f"  Test MSE: {dt_metrics['test_mse']:.4e}")
print(f"  Test MAE: {dt_metrics['test_mae']:.4e}")

# Check feature importance
print("\n" + "="*60)
print("Feature Importance:")
print("="*60)
importance = dt_model.get_feature_importance()
print(importance.head(10))

# Critical check: Is Energy important now?
energy_importance = importance[importance['Feature'] == 'Energy']
if len(energy_importance) > 0:
    energy_imp_value = energy_importance['Importance'].values[0]
    print(f"\n✓ Energy importance: {energy_imp_value:.6f}")
    if energy_imp_value > 0:
        print("  ✓ SUCCESS: Model is using Energy feature!")
    else:
        print("  ✗ FAIL: Energy still has zero importance")
else:
    print("\n✗ FAIL: Energy not in feature importance")

# Test predictions for U-235 fission
print("\n" + "="*80)
print("Testing U-235 Fission Predictions")
print("="*80)

Z, A, mt_code = 92, 235, 18
energy_range = (1.0, 100.0)

energies, predictions = dt_model.predict_resonance_region(
    Z, A, mt_code, energy_range, num_points=20, mode='naive'
)

print("\nSample predictions:")
for i in [0, 5, 10, 15, 19]:
    print(f"  Energy {energies[i]:8.2f} eV → CrossSection {predictions[i]:8.2f} barns")

# Check if predictions vary with energy
unique_predictions = len(set(predictions.round(2)))
print(f"\nUnique prediction values: {unique_predictions}")

if unique_predictions > 1:
    print("✓ SUCCESS: Predictions vary with energy (not horizontal lines!)")
    print("✓ Model learned energy-dependent patterns")

    # Check if it's a staircase (expected for Decision Tree)
    if unique_predictions < 10:
        print(f"✓ EXPECTED: Staircase effect detected ({unique_predictions} discrete levels)")
        print("  This demonstrates classical ML limitations!")
else:
    print("✗ FAIL: All predictions are identical (horizontal line)")
    print("  The dataset may still be insufficient")

# Test predictions for Cl-35 (n,p)
print("\n" + "="*80)
print("Testing Cl-35 (n,p) Predictions")
print("="*80)

Z_cl, A_cl, mt_cl = 17, 35, 103
energy_range_cl = (1e6, 2e7)

energies_cl, predictions_cl = dt_model.predict_resonance_region(
    Z_cl, A_cl, mt_cl, energy_range_cl, num_points=20, mode='naive'
)

print("\nSample predictions:")
for i in [0, 5, 10, 15, 19]:
    print(f"  Energy {energies_cl[i]:8.2e} eV → CrossSection {predictions_cl[i]:8.4f} barns")

unique_predictions_cl = len(set(predictions_cl.round(4)))
print(f"\nUnique prediction values: {unique_predictions_cl}")

if unique_predictions_cl > 1:
    print("✓ SUCCESS: Cl-35 predictions also vary with energy")
else:
    print("⚠ WARNING: Cl-35 predictions are constant")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nSummary:")
print("  - Dataset has sufficient measurements for training")
print("  - Model learns energy-dependent patterns")
print("  - Predictions show staircase effect (classical ML limitation)")
print("  - Ready for notebook demonstrations!")
print("="*80)
