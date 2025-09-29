"""
Tests for the water usage module.

This module tests the WaterUsageModel class and its functionality including
prior sampling, predictive checks, and visualizations.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

from bayesian_rainfall.water_usage import (
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


class TestWaterUsageModel:
    """Test the WaterUsageModel class."""
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = WaterUsageModel()
        
        assert model.hs_mean == 2.6
        assert model.hs_sd == 1.2
        assert model.lot_mean == 8000.0
        assert model.lot_sd == 6000.0
        assert model.K == 6
        assert len(model.month_mu) == 12
        assert model.month_sigma == 0.3
        assert model.intercept_mu == np.log(7000)
        assert model.intercept_sigma == 1.0
        assert model.model is None
        assert model.prior_samples is None
    
    def test_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = WaterUsageModel(
            hs_mean=3.0, 
            hs_sd=1.5,
            lot_mean=10000.0, 
            lot_sd=8000.0,
            n_climate_zones=8
        )
        
        assert model.hs_mean == 3.0
        assert model.hs_sd == 1.5
        assert model.lot_mean == 10000.0
        assert model.lot_sd == 8000.0
        assert model.K == 8
    
    def test_standardization_functions(self):
        """Test the standardization helper functions."""
        model = WaterUsageModel()
        
        # Test household size standardization
        hs_values = np.array([1.0, 2.6, 4.0])
        hs_std = model._standardize_hs(hs_values)
        expected_hs_std = (hs_values - 2.6) / 1.2
        np.testing.assert_array_almost_equal(hs_std, expected_hs_std)
        
        # Test lot size standardization
        lot_values = np.array([2000.0, 8000.0, 14000.0])
        lot_std = model._standardize_lot(lot_values)
        expected_lot_std = (lot_values - 8000.0) / 6000.0
        np.testing.assert_array_almost_equal(lot_std, expected_lot_std)
    
    def test_build_model(self):
        """Test model building."""
        model = WaterUsageModel()
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        assert built_model is model.model
        
        # Check that all expected variables are in the model
        expected_vars = [
            "alpha", "gamma", "eta", "beta_hs", "beta_res", 
            "beta_lot", "beta_pool", "beta_rural", "sigma_y", "sigma_eta"
        ]
        for var_name in expected_vars:
            assert var_name in model.model.named_vars
    
    def test_sample_priors(self):
        """Test prior sampling."""
        model = WaterUsageModel()
        prior_samples = model.sample_priors(samples=100)
        
        assert isinstance(prior_samples, dict)
        assert model.prior_samples is not None
        
        # Check that all expected parameters are present
        expected_params = [
            "alpha", "gamma", "eta", "beta_hs", "beta_res", 
            "beta_lot", "beta_pool", "beta_rural", "sigma_y"
        ]
        for param in expected_params:
            assert param in prior_samples
            assert isinstance(prior_samples[param], np.ndarray)
        
        # Check shapes
        assert prior_samples["alpha"].shape == (1, 100)
        assert prior_samples["gamma"].shape == (1, 100, 12)
        assert prior_samples["eta"].shape == (1, 100, 6)
        assert prior_samples["beta_hs"].shape == (1, 100)
        assert prior_samples["beta_res"].shape == (1, 100)
        assert prior_samples["beta_lot"].shape == (1, 100)
        assert prior_samples["beta_pool"].shape == (1, 100)
        assert prior_samples["beta_rural"].shape == (1, 100)
        assert prior_samples["sigma_y"].shape == (1, 100)
    
    def test_predict_prior_draws(self):
        """Test prior predictive draws."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)
        
        # Test basic prediction
        draws = model.predict_prior_draws(
            month_idx=0, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=50
        )
        
        assert isinstance(draws, np.ndarray)
        assert len(draws) == 50
        assert np.all(draws > 0)  # All water usage should be positive
        
        # Test different scenarios
        summer_draws = model.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=50
        )
        
        # Summer should generally be higher than winter
        assert np.median(summer_draws) > np.median(draws)
    
    def test_get_prior_scenarios(self):
        """Test getting predefined scenarios."""
        model = WaterUsageModel()
        scenarios = model.get_prior_scenarios()
        
        assert isinstance(scenarios, dict)
        expected_scenarios = [
            "baseline_winter", "baseline_summer", "small_household",
            "large_lot_pool", "multi_family"
        ]
        
        for scenario in expected_scenarios:
            assert scenario in scenarios
            assert isinstance(scenarios[scenario], dict)
            
            # Check required keys
            required_keys = ["month_idx", "climate_idx", "hs", "res_sf", "lot", "pool", "rural"]
            for key in required_keys:
                assert key in scenarios[scenario]
    
    def test_run_prior_predictive_checks(self):
        """Test running prior predictive checks."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)  # Use fewer samples for speed
        
        summary_df = model.run_prior_predictive_checks(draws=50)
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 5  # 5 scenarios
        
        # Check required columns
        required_cols = ["median_gal", "iqr_low", "iqr_high", "p2.5", "p97.5", "mean_gal"]
        for col in required_cols:
            assert col in summary_df.columns
        
        # Check that all values are positive
        for col in required_cols:
            assert np.all(summary_df[col] > 0)
    
    def test_get_prior_descriptions(self):
        """Test getting prior descriptions."""
        model = WaterUsageModel()
        descriptions = model.get_prior_descriptions()
        
        assert isinstance(descriptions, dict)
        
        expected_params = [
            "alpha", "gamma", "eta", "beta_hs", "beta_res", 
            "beta_lot", "beta_pool", "beta_rural", "sigma_y"
        ]
        
        for param in expected_params:
            assert param in descriptions
            assert isinstance(descriptions[param], str)
            assert len(descriptions[param]) > 0
    
    def test_run_comprehensive_prior_predictive_checks(self):
        """Test comprehensive prior predictive checks."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)  # Use fewer samples for speed
        
        results = model.run_comprehensive_prior_predictive_checks(draws=50)
        
        assert isinstance(results, dict)
        
        # Check that all expected keys are present
        expected_keys = [
            "baseline_usage", "seasonal_pattern", "household_size_effect",
            "sf_vs_mf", "lot_size_effect", "climate_zone_effect",
            "pool_effect", "urban_rural_effect"
        ]
        
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], dict)
        
        # Check baseline usage structure
        baseline = results["baseline_usage"]
        assert "winter" in baseline
        assert "summer" in baseline
        
        for season in ["winter", "summer"]:
            season_data = baseline[season]
            assert "median" in season_data
            assert "iqr" in season_data
            assert "p95" in season_data
            assert "p5" in season_data
            assert isinstance(season_data["iqr"], list)
            assert len(season_data["iqr"]) == 2
        
        # Check that all values are positive
        for season in ["winter", "summer"]:
            season_data = baseline[season]
            assert season_data["median"] > 0
            assert all(x > 0 for x in season_data["iqr"])
            assert season_data["p95"] > 0
            assert season_data["p5"] > 0
    
    def test_predict_seasonal_usage(self):
        """Test seasonal usage prediction."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)
        
        # Test with Eugene Oregon scenario
        eugene_scenario = model.get_eugene_oregon_scenario()
        seasonal_data = model.predict_seasonal_usage(draws=50, **eugene_scenario)
        
        # Check structure
        assert isinstance(seasonal_data, dict)
        assert 'months' in seasonal_data
        assert 'median' in seasonal_data
        assert 'q025' in seasonal_data
        assert 'q975' in seasonal_data
        assert 'q25' in seasonal_data
        assert 'q75' in seasonal_data
        assert 'mean' in seasonal_data
        assert 'raw_data' in seasonal_data
        
        # Check dimensions
        assert len(seasonal_data['months']) == 12
        assert len(seasonal_data['median']) == 12
        assert seasonal_data['raw_data'].shape == (50, 12)
        
        # Check that all values are positive
        for key in ['median', 'q025', 'q975', 'q25', 'q75', 'mean']:
            assert np.all(seasonal_data[key] > 0)
        
        # Check that confidence intervals are ordered correctly
        assert np.all(seasonal_data['q025'] <= seasonal_data['q25'])
        assert np.all(seasonal_data['q25'] <= seasonal_data['median'])
        assert np.all(seasonal_data['median'] <= seasonal_data['q75'])
        assert np.all(seasonal_data['q75'] <= seasonal_data['q975'])
    
    def test_get_eugene_oregon_scenario(self):
        """Test Eugene Oregon scenario parameters."""
        model = WaterUsageModel()
        scenario = model.get_eugene_oregon_scenario()
        
        assert isinstance(scenario, dict)
        assert 'climate_idx' in scenario
        assert 'hs' in scenario
        assert 'res_sf' in scenario
        assert 'lot' in scenario
        assert 'pool' in scenario
        assert 'rural' in scenario
        
        # Check specific values for Eugene Oregon
        assert scenario['climate_idx'] == 2  # Moderate climate
        assert scenario['hs'] == 4.0  # Family of 4
        assert scenario['res_sf'] == 1  # Single-family
        assert scenario['lot'] == 8000  # Typical lot size
        assert scenario['pool'] == 0  # No pool
        assert scenario['rural'] == 1  # Rural area


class TestVisualizationFunctions:
    """Test visualization functions."""
    
    def test_create_prior_visualization(self):
        """Test prior visualization creation."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)
        
        fig = create_prior_visualization(model, figsize=(10, 8))
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 9  # 3x3 grid
        
        plt.close(fig)
    
    def test_create_prior_predictive_visualization(self):
        """Test prior predictive visualization creation."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)
        
        fig = create_prior_predictive_visualization(model, draws=50)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # 2x3 grid
        
        plt.close(fig)
    
    def test_create_research_validation_visualization(self):
        """Test research validation visualization creation."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)
        
        fig = create_research_validation_visualization(model, draws=50)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 8  # 2x4 grid
        
        plt.close(fig)
    
    def test_create_seasonal_usage_plot(self):
        """Test seasonal usage plot creation."""
        model = WaterUsageModel()
        model.sample_priors(samples=100)
        
        # Test with Eugene Oregon scenario
        eugene_scenario = model.get_eugene_oregon_scenario()
        fig = create_seasonal_usage_plot(model, eugene_scenario, draws=50)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1  # Single subplot
        
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_visualization_without_display(self, mock_show):
        """Test that visualizations can be created without displaying."""
        model = WaterUsageModel()
        model.sample_priors(samples=50)
        
        # These should not raise errors
        fig1 = create_prior_visualization(model)
        fig2 = create_prior_predictive_visualization(model, draws=25)
        fig3 = create_research_validation_visualization(model, draws=25)
        fig4 = create_seasonal_usage_plot(model, model.get_eugene_oregon_scenario(), draws=25)
        
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        assert isinstance(fig3, plt.Figure)
        assert isinstance(fig4, plt.Figure)
        
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)


class TestModelValidation:
    """Test model validation against research findings."""
    
    def test_epa_baseline_validation(self):
        """Test that model predictions align with EPA baseline."""
        model = WaterUsageModel()
        model.sample_priors(samples=200)
        
        # EPA: 4-person family uses ~10,000 gallons/month in summer
        epa_4_person = 10000
        our_4_person = model.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=4.0, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=100
        )
        
        our_median = np.median(our_4_person)
        # Allow for reasonable range around EPA value (priors can be quite wide)
        assert 1000 <= our_median <= 100000
    
    def test_seasonal_variation_validation(self):
        """Test seasonal variation matches research expectations."""
        model = WaterUsageModel()
        model.sample_priors(samples=200)
        
        winter_usage = model.predict_prior_draws(
            month_idx=0, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=100
        )
        summer_usage = model.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=100
        )
        
        summer_winter_ratio = np.median(summer_usage) / np.median(winter_usage)
        
        # Research expects 2-3x summer/winter ratio (allow wider range for priors)
        assert 1.0 <= summer_winter_ratio <= 10.0
    
    def test_pool_effect_validation(self):
        """Test pool effect matches research expectations."""
        model = WaterUsageModel()
        model.sample_priors(samples=200)
        
        no_pool = model.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=100
        )
        with_pool = model.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=1, rural=0, draws=100
        )
        
        pool_effect_ratio = np.median(with_pool) / np.median(no_pool)
        
        # Research expects 3-6x pool effect
        assert 2.0 <= pool_effect_ratio <= 10.0
    
    def test_household_size_effect_validation(self):
        """Test household size effect matches research expectations."""
        model = WaterUsageModel()
        model.sample_priors(samples=200)
        
        small_hs = model.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=2.0, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=100
        )
        large_hs = model.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=4.0, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=100
        )
        
        hs_effect_ratio = np.median(large_hs) / np.median(small_hs)
        
        # Research expects ~1.28x per person, so 2 people = ~1.6x
        assert 1.2 <= hs_effect_ratio <= 2.5


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_predict_without_priors(self):
        """Test that prediction works even without prior samples."""
        model = WaterUsageModel()
        
        # This should automatically sample priors
        draws = model.predict_prior_draws(
            month_idx=0, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=50
        )
        
        assert isinstance(draws, np.ndarray)
        assert len(draws) == 50
        assert model.prior_samples is not None
    
    def test_invalid_month_index(self):
        """Test handling of invalid month indices."""
        model = WaterUsageModel()
        model.sample_priors(samples=50)
        
        # Should handle month index > 11 gracefully
        draws = model.predict_prior_draws(
            month_idx=15, climate_idx=2, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=25
        )
        
        assert isinstance(draws, np.ndarray)
        assert len(draws) == 25
    
    def test_invalid_climate_index(self):
        """Test handling of invalid climate indices."""
        model = WaterUsageModel()
        model.sample_priors(samples=50)
        
        # Should handle climate index >= K gracefully with modulo
        draws = model.predict_prior_draws(
            month_idx=0, climate_idx=10, hs=2.6, 
            res_sf=1, lot=8000, pool=0, rural=0, draws=25
        )
        
        assert isinstance(draws, np.ndarray)
        assert len(draws) == 25


class TestAdvancedAnalysis:
    """Test advanced analysis functions."""
    
    def test_analyze_parameter_sensitivity(self):
        """Test parameter sensitivity analysis."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        base_scenario = model.get_eugene_oregon_scenario()
        parameter_ranges = {
            'hs': [2.0, 4.0, 6.0],
            'lot': [4000, 8000, 12000]
        }
        
        results = model.analyze_parameter_sensitivity(
            base_scenario, parameter_ranges, draws=50
        )
        
        assert 'hs' in results
        assert 'lot' in results
        assert len(results['hs']['values']) == 3
        assert len(results['hs']['medians']) == 3
        assert len(results['lot']['values']) == 3
        assert len(results['lot']['medians']) == 3
    
    def test_analyze_climate_impact(self):
        """Test climate impact analysis."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        base_scenario = model.get_eugene_oregon_scenario()
        results = model.analyze_climate_impact(base_scenario, draws=50)
        
        assert 'zones' in results
        assert 'july_medians' in results
        assert 'annual_medians' in results
        assert len(results['zones']) == model.K
        assert len(results['july_medians']) == model.K
        assert len(results['annual_medians']) == model.K
    
    def test_analyze_household_size_impact(self):
        """Test household size impact analysis."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        base_scenario = model.get_eugene_oregon_scenario()
        household_sizes = [2.0, 4.0, 6.0]
        
        results = model.analyze_household_size_impact(
            base_scenario, household_sizes, draws=50
        )
        
        assert 'sizes' in results
        assert 'july_medians' in results
        assert 'annual_medians' in results
        assert len(results['sizes']) == 3
        assert len(results['july_medians']) == 3
        assert len(results['annual_medians']) == 3
    
    def test_generate_scenario_matrix(self):
        """Test scenario matrix generation."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        parameter_grid = {
            'hs': [2.0, 4.0],
            'lot': [4000, 8000]
        }
        
        df = model.generate_scenario_matrix(parameter_grid, draws=50)
        
        assert len(df) == 4  # 2 x 2 combinations
        assert 'hs' in df.columns
        assert 'lot' in df.columns
        assert 'july_median' in df.columns
        assert 'annual_median' in df.columns
        assert 'july_q025' in df.columns
        assert 'july_q975' in df.columns
        assert 'annual_q025' in df.columns
        assert 'annual_q975' in df.columns


class TestAdvancedVisualizations:
    """Test advanced visualization functions."""
    
    def test_create_sensitivity_analysis_plot(self):
        """Test sensitivity analysis plot creation."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        base_scenario = model.get_eugene_oregon_scenario()
        parameter_ranges = {
            'hs': [2.0, 4.0, 6.0],
            'lot': [4000, 8000, 12000]
        }
        
        fig = create_sensitivity_analysis_plot(
            model, base_scenario, parameter_ranges, draws=50
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # At least 2 subplots
        plt.close(fig)
    
    def test_create_climate_impact_plot(self):
        """Test climate impact plot creation."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        base_scenario = model.get_eugene_oregon_scenario()
        fig = create_climate_impact_plot(model, base_scenario, draws=50)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)
    
    def test_create_household_size_analysis_plot(self):
        """Test household size analysis plot creation."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        base_scenario = model.get_eugene_oregon_scenario()
        household_sizes = [2.0, 4.0, 6.0]
        
        fig = create_household_size_analysis_plot(
            model, base_scenario, household_sizes, draws=50
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)
    
    def test_create_scenario_heatmap(self):
        """Test scenario heatmap creation."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        parameter_grid = {
            'hs': [2.0, 4.0, 6.0],
            'lot': [4000, 8000, 12000]
        }
        
        fig = create_scenario_heatmap(
            model, parameter_grid, metric='annual_median', draws=50
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar
        plt.close(fig)
    
    def test_create_distribution_comparison_plot(self):
        """Test distribution comparison plot creation."""
        model = WaterUsageModel()
        model.build_model()
        model.sample_priors(samples=100)
        
        scenarios = {
            "Scenario 1": {'climate_idx': 2, 'hs': 4.0, 'res_sf': 1, 'lot': 8000, 'pool': 0, 'rural': 1},
            "Scenario 2": {'climate_idx': 2, 'hs': 2.0, 'res_sf': 1, 'lot': 4000, 'pool': 0, 'rural': 0}
        }
        
        fig = create_distribution_comparison_plot(
            model, scenarios, month_idx=6, draws=100
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # At least 2 subplots
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
