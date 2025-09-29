"""
Water Usage Module

This module provides Bayesian models for residential water usage prediction.
"""

from .model import (
    WaterUsageModel,
    create_prior_visualization,
    create_prior_predictive_visualization,
    create_research_validation_visualization,
    create_seasonal_usage_plot,
    create_sensitivity_analysis_plot,
    create_climate_impact_plot,
    create_household_size_analysis_plot,
    create_scenario_heatmap,
    create_distribution_comparison_plot
)

__all__ = [
    "WaterUsageModel",
    "create_prior_visualization", 
    "create_prior_predictive_visualization",
    "create_research_validation_visualization",
    "create_seasonal_usage_plot",
    "create_sensitivity_analysis_plot",
    "create_climate_impact_plot",
    "create_household_size_analysis_plot",
    "create_scenario_heatmap",
    "create_distribution_comparison_plot"
]
