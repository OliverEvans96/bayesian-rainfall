#!/usr/bin/env python3
"""
Demonstration script for the Bayesian Water Usage Model.

This script shows the key functionality of the water usage model including
prior sampling, predictive checks, and validation against research findings.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from bayesian_rainfall.water_usage import (
    WaterUsageModel, 
    create_prior_visualization,
    create_prior_predictive_visualization
)


def main():
    """Demonstrate the water usage model functionality."""
    print("🌊 Bayesian Water Usage Model Demonstration")
    print("=" * 50)
    
    # Initialize model
    print("\n1. Initializing model...")
    model = WaterUsageModel()
    print(f"   ✓ Climate zones: {model.K}")
    print(f"   ✓ Standardization: HS({model.hs_mean:.1f}±{model.hs_sd:.1f}), Lot({model.lot_mean:.0f}±{model.lot_sd:.0f})")
    
    # Build model and sample priors
    print("\n2. Building PyMC model and sampling priors...")
    model.build_model()
    prior_samples = model.sample_priors(samples=500)
    print(f"   ✓ Sampled {len(prior_samples['alpha'])} prior samples")
    
    # Show prior descriptions
    print("\n3. Prior distributions:")
    descriptions = model.get_prior_descriptions()
    for param, desc in descriptions.items():
        print(f"   • {param}: {desc}")
    
    # Run prior predictive checks
    print("\n4. Running prior predictive checks...")
    summary_df = model.run_prior_predictive_checks(draws=200)
    print("\n   Prior Predictive Summary (gallons/month):")
    print(summary_df.round(0))
    
    # Validation against research
    print("\n5. Validation against research findings:")
    
    # EPA baseline validation
    epa_4_person = 10000
    our_4_person = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=4.0, 
        res_sf=1, lot=8000, pool=0, rural=0, draws=200
    )
    our_median = np.median(our_4_person)
    print(f"   • EPA 4-person family (summer): {epa_4_person:,} gal/month")
    print(f"   • Our model median: {our_median:,.0f} gal/month")
    print(f"   • Difference: {((our_median - epa_4_person) / epa_4_person * 100):+.1f}%")
    
    # Seasonal variation
    winter_usage = model.predict_prior_draws(
        month_idx=0, climate_idx=2, hs=2.6, 
        res_sf=1, lot=8000, pool=0, rural=0, draws=200
    )
    summer_usage = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=2.6, 
        res_sf=1, lot=8000, pool=0, rural=0, draws=200
    )
    summer_winter_ratio = np.median(summer_usage) / np.median(winter_usage)
    print(f"   • Summer/Winter ratio: {summer_winter_ratio:.1f}x (expected: 2-3x)")
    
    # Pool effect
    no_pool = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=2.6, 
        res_sf=1, lot=8000, pool=0, rural=0, draws=200
    )
    with_pool = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=2.6, 
        res_sf=1, lot=8000, pool=1, rural=0, draws=200
    )
    pool_effect_ratio = np.median(with_pool) / np.median(no_pool)
    print(f"   • Pool effect: {pool_effect_ratio:.1f}x (expected: 3-6x)")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    print("   • Prior distributions plot...")
    fig1 = create_prior_visualization(model, figsize=(12, 8))
    fig1.suptitle("Bayesian Water Usage Model: Prior Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig("water_usage_priors.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   • Prior predictive checks plot...")
    fig2 = create_prior_predictive_visualization(model, draws=200)
    fig2.suptitle("Prior Predictive Checks: Water Usage Scenarios", fontsize=14)
    plt.tight_layout()
    plt.savefig("water_usage_predictive_checks.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✓ Visualizations saved as PNG files")
    
    # Scenario comparison
    print("\n7. Scenario comparison:")
    scenarios = model.get_prior_scenarios()
    print(f"   Available scenarios: {list(scenarios.keys())}")
    
    # Show a few example predictions
    print("\n8. Example predictions:")
    for name, sc in list(scenarios.items())[:3]:
        draws = model.predict_prior_draws(**sc, draws=100)
        median_usage = np.median(draws)
        print(f"   • {name.replace('_', ' ').title()}: {median_usage:.0f} gal/month")
    
    print("\n" + "=" * 50)
    print("✅ Demonstration complete!")
    print("   The water usage model is ready for:")
    print("   • Data integration and posterior analysis")
    print("   • Water usage prediction for new households")
    print("   • Policy analysis and conservation planning")


if __name__ == "__main__":
    main()
