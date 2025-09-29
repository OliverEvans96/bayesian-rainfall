"""
Bayesian Water Usage Model

This module provides a Bayesian model for residential water usage prediction
based on household characteristics, seasonal patterns, and climate zones.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class WaterUsageModel:
    """
    Bayesian model for residential water usage prediction.
    
    This model uses log-linear regression with informative priors based on
    U.S. water usage research and data.
    """
    
    def __init__(self, 
                 hs_mean: float = 2.6, 
                 hs_sd: float = 1.2,
                 lot_mean: float = 8000.0, 
                 lot_sd: float = 6000.0,
                 n_climate_zones: int = 6):
        """
        Initialize the water usage model with standardization parameters.
        
        Parameters
        ----------
        hs_mean, hs_sd : float
            Mean and standard deviation for household size standardization
        lot_mean, lot_sd : float
            Mean and standard deviation for lot size standardization
        n_climate_zones : int
            Number of climate zones to model
        """
        self.hs_mean = hs_mean
        self.hs_sd = hs_sd
        self.lot_mean = lot_mean
        self.lot_sd = lot_sd
        self.K = n_climate_zones
        
        # Month baseline means (log-scale multiplicative offsets)
        # Jan..Dec: small in winter, peak in summer (values based on prior research)
        self.month_mu = np.array([
            0.00,  # Jan
            0.00,  # Feb
            0.10,  # Mar
            0.20,  # Apr
            0.40,  # May
            0.50,  # Jun
            0.90,  # Jul
            0.80,  # Aug
            0.50,  # Sep
            0.20,  # Oct
            0.00,  # Nov
            0.00   # Dec
        ])
        self.month_sigma = 0.30  # uncertainty around month means
        
        # Intercept prior (log monthly gallons)
        self.intercept_mu = np.log(7000)
        self.intercept_sigma = 1.0
        
        self.model = None
        self.prior_samples = None
        
    def _standardize_hs(self, x: np.ndarray) -> np.ndarray:
        """Standardize household size."""
        return (x - self.hs_mean) / self.hs_sd
    
    def _standardize_lot(self, x: np.ndarray) -> np.ndarray:
        """Standardize lot size."""
        return (x - self.lot_mean) / self.lot_sd
    
    def build_model(self) -> pm.Model:
        """
        Build the PyMC model with priors.
        
        Returns
        -------
        pm.Model
            The PyMC model with all priors defined
        """
        with pm.Model() as model:
            # Global intercept
            alpha = pm.Normal("alpha", mu=self.intercept_mu, sigma=self.intercept_sigma)
            
            # Month effects: informative means with moderate SD
            gamma = pm.Normal("gamma", mu=self.month_mu, sigma=self.month_sigma, shape=12)
            
            # Climate-zone varying intercepts
            sigma_eta = pm.HalfNormal("sigma_eta", sigma=0.5)
            eta = pm.Normal("eta", mu=0.0, sigma=sigma_eta, shape=self.K)
            
            # Coefficients for standardized predictors
            # hs: per-person effect (std units) ~ N(0.25, 0.1)
            beta_hs = pm.Normal("beta_hs", mu=0.25, sigma=0.1)
            # res: single-family vs multi-family (binary) ~ N(0.2, 0.1)
            beta_res = pm.Normal("beta_res", mu=0.2, sigma=0.1)
            # lot: standardized lot size effect ~ N(0.3, 0.2)
            beta_lot = pm.Normal("beta_lot", mu=0.3, sigma=0.2)
            # out: pool effect (binary), large positive ~ N(1.5, 0.5)
            beta_pool = pm.Normal("beta_pool", mu=1.5, sigma=0.5)
            # urb: rural vs urban effect ~ N(0.2, 0.2)
            beta_rural = pm.Normal("beta_rural", mu=0.2, sigma=0.2)
            
            # Observation noise on log-scale
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.5)
        
        self.model = model
        return model
    
    def sample_priors(self, samples: int = 2000) -> Dict[str, np.ndarray]:
        """
        Sample from prior distributions.
        
        Parameters
        ----------
        samples : int
            Number of prior samples to draw
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing prior samples for all parameters
        """
        if self.model is None:
            self.build_model()
            
        with self.model:
            prior_samples = pm.sample_prior_predictive(
                samples=samples, 
                var_names=[
                    "alpha", "gamma", "eta", "beta_hs", "beta_res", 
                    "beta_lot", "beta_pool", "beta_rural", "sigma_y"
                ]
            )
        
        # Convert InferenceData to dictionary format
        prior_dict = {}
        for var_name in ["alpha", "gamma", "eta", "beta_hs", "beta_res", 
                        "beta_lot", "beta_pool", "beta_rural", "sigma_y"]:
            prior_dict[var_name] = prior_samples.prior[var_name].values
        
        self.prior_samples = prior_dict
        return prior_dict
    
    def predict_prior_draws(self, 
                           month_idx: int, 
                           climate_idx: int, 
                           hs: float, 
                           res_sf: int, 
                           lot: float, 
                           pool: int, 
                           rural: int, 
                           draws: int = 2000) -> np.ndarray:
        """
        Generate prior predictive draws for a specific scenario.
        
        Parameters
        ----------
        month_idx : int
            Month index (0-11)
        climate_idx : int
            Climate zone index (0 to K-1)
        hs : float
            Household size (persons)
        res_sf : int
            1 if single-family, 0 if multi-family
        lot : float
            Lot size in sqft
        pool : int
            1 if pool present, 0 otherwise
        rural : int
            1 if rural, 0 if urban
        draws : int
            Number of draws to generate
            
        Returns
        -------
        np.ndarray
            Array of predicted monthly water usage in gallons
        """
        if self.prior_samples is None:
            self.sample_priors()
            
        # Standardize continuous features
        hs_s = self._standardize_hs(hs)
        lot_s = self._standardize_lot(lot)
        
        # Pick random prior draws
        n_samples = len(self.prior_samples["alpha"])
        if draws > n_samples:
            # If we need more draws than available, sample with replacement
            idx = np.random.choice(n_samples, size=draws, replace=True)
        else:
            idx = np.random.choice(n_samples, size=draws, replace=False)
        
        # Extract parameter draws (flatten first to get 1D arrays)
        alpha_d = self.prior_samples["alpha"].flatten()[idx]
        # Handle month index with modulo for invalid indices
        gamma_d = self.prior_samples["gamma"].reshape(-1, 12)[idx, month_idx % 12]
        eta_d = self.prior_samples["eta"].reshape(-1, self.K)[idx, climate_idx % self.K]
        bh = self.prior_samples["beta_hs"].flatten()[idx]
        br = self.prior_samples["beta_res"].flatten()[idx]
        bl = self.prior_samples["beta_lot"].flatten()[idx]
        bp = self.prior_samples["beta_pool"].flatten()[idx]
        brr = self.prior_samples["beta_rural"].flatten()[idx]
        sy = self.prior_samples["sigma_y"].flatten()[idx]
        
        # Calculate mean log usage
        mu = (alpha_d + gamma_d + eta_d +
              bh * hs_s +
              br * res_sf +
              bl * lot_s +
              bp * pool +
              brr * rural)
        
        # Draw observed log-use with noise
        y_log = np.random.normal(loc=mu, scale=sy)
        y_gal = np.exp(y_log)
        return y_gal
    
    def get_prior_scenarios(self) -> Dict[str, Dict]:
        """
        Get predefined scenarios for prior predictive checks.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary of scenario names and their parameters
        """
        return {
            "baseline_winter": dict(month_idx=0, climate_idx=2, hs=2.6, res_sf=1, lot=8000, pool=0, rural=0),  # Jan
            "baseline_summer": dict(month_idx=6, climate_idx=2, hs=2.6, res_sf=1, lot=8000, pool=0, rural=0),  # Jul
            "small_household": dict(month_idx=6, climate_idx=2, hs=1.5, res_sf=0, lot=0, pool=0, rural=1),
            "large_lot_pool": dict(month_idx=6, climate_idx=0, hs=4.0, res_sf=1, lot=15000, pool=1, rural=1),
            "multi_family": dict(month_idx=6, climate_idx=3, hs=2.6, res_sf=0, lot=2000, pool=0, rural=0)
        }
    
    def run_prior_predictive_checks(self, draws: int = 2000) -> pd.DataFrame:
        """
        Run prior predictive checks across all scenarios.
        
        Parameters
        ----------
        draws : int
            Number of draws per scenario
            
        Returns
        -------
        pd.DataFrame
            Summary statistics for each scenario
        """
        scenarios = self.get_prior_scenarios()
        summary_rows = []
        
        for name, sc in scenarios.items():
            draws_array = self.predict_prior_draws(**sc, draws=draws)
            summary_rows.append({
                "scenario": name,
                "median_gal": np.median(draws_array),
                "iqr_low": np.percentile(draws_array, 25),
                "iqr_high": np.percentile(draws_array, 75),
                "p2.5": np.percentile(draws_array, 2.5),
                "p97.5": np.percentile(draws_array, 97.5),
                "mean_gal": np.mean(draws_array)
            })
        
        return pd.DataFrame(summary_rows).set_index("scenario")
    
    def get_prior_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of the prior distributions.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping parameter names to their prior descriptions
        """
        return {
            "alpha": "Global intercept ~ N(log(7000), 1.0) - baseline monthly usage",
            "gamma": "Month effects ~ N(month_mu, 0.3) - seasonal patterns",
            "eta": "Climate zone effects ~ N(0, sigma_eta) - regional differences",
            "beta_hs": "Household size effect ~ N(0.25, 0.1) - per-person scaling",
            "beta_res": "Residence type effect ~ N(0.2, 0.1) - SF vs MF difference",
            "beta_lot": "Lot size effect ~ N(0.3, 0.2) - outdoor irrigation impact",
            "beta_pool": "Pool effect ~ N(1.5, 0.5) - large outdoor water use",
            "beta_rural": "Urban/rural effect ~ N(0.2, 0.2) - location differences",
            "sigma_y": "Observation noise ~ HalfNormal(0.5) - residual variation"
        }
    
    def run_comprehensive_prior_predictive_checks(self, draws: int = 2000) -> Dict[str, Dict]:
        """
        Run comprehensive prior predictive checks as described in water-use-priors.txt.
        
        Parameters
        ----------
        draws : int
            Number of draws per scenario
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary containing detailed results for each check
        """
        if self.prior_samples is None:
            self.sample_priors()
        
        results = {}
        
        # 1. Baseline monthly usage (2-4 person household, SF, moderate climate, no pool)
        baseline_winter = self.predict_prior_draws(
            month_idx=0, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        baseline_summer = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        
        results["baseline_usage"] = {
            "winter": {
                "median": np.median(baseline_winter),
                "iqr": [np.percentile(baseline_winter, 25), np.percentile(baseline_winter, 75)],
                "p95": np.percentile(baseline_winter, 95),
                "p5": np.percentile(baseline_winter, 5)
            },
            "summer": {
                "median": np.median(baseline_summer),
                "iqr": [np.percentile(baseline_summer, 25), np.percentile(baseline_summer, 75)],
                "p95": np.percentile(baseline_summer, 95),
                "p5": np.percentile(baseline_summer, 5)
            }
        }
        
        # 2. Seasonal pattern validation
        seasonal_ratio = np.median(baseline_summer) / np.median(baseline_winter)
        results["seasonal_pattern"] = {
            "summer_winter_ratio": seasonal_ratio,
            "expected_range": [1.5, 2.5],
            "passes": 1.5 <= seasonal_ratio <= 2.5
        }
        
        # 3. Household size effect (3 vs 4 person)
        hs_3 = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        hs_4 = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=4.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        hs_5 = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=5.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        
        hs_effect_4_3 = np.median(hs_4) / np.median(hs_3)
        hs_effect_5_3 = np.median(hs_5) / np.median(hs_3)
        
        results["household_size_effect"] = {
            "effect_4_3": hs_effect_4_3,
            "effect_5_3": hs_effect_5_3,
            "expected_4_3": 1.28,  # exp(0.25)
            "expected_5_3": 1.6,   # exp(0.25 * 2)
            "passes_4_3": 1.2 <= hs_effect_4_3 <= 1.4,
            "passes_5_3": 1.5 <= hs_effect_5_3 <= 1.8
        }
        
        # 4. SF vs MF comparison
        sf_usage = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        mf_usage = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=0, lot=8000, pool=0, rural=0, draws=draws
        )
        
        sf_mf_ratio = np.median(sf_usage) / np.median(mf_usage)
        results["sf_vs_mf"] = {
            "sf_mf_ratio": sf_mf_ratio,
            "expected_ratio": 1.22,  # exp(0.2)
            "passes": 1.1 <= sf_mf_ratio <= 1.4
        }
        
        # 5. Lot size effect (1σ and 2σ above mean)
        lot_small = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=2000, pool=0, rural=0, draws=draws
        )
        lot_medium = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        lot_large = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=14000, pool=0, rural=0, draws=draws
        )
        lot_very_large = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=20000, pool=0, rural=0, draws=draws
        )
        
        lot_effect_1std = np.median(lot_large) / np.median(lot_medium)
        lot_effect_2std = np.median(lot_very_large) / np.median(lot_medium)
        
        results["lot_size_effect"] = {
            "effect_1std": lot_effect_1std,
            "effect_2std": lot_effect_2std,
            "expected_1std": 1.35,  # exp(0.3)
            "expected_2std": 1.82,  # exp(0.3 * 2)
            "passes_1std": 1.2 <= lot_effect_1std <= 1.6,
            "passes_2std": 1.5 <= lot_effect_2std <= 2.5
        }
        
        # 6. Climate zone effects
        climate_cool = self.predict_prior_draws(
            month_idx=6, climate_idx=0, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        climate_moderate = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        climate_hot = self.predict_prior_draws(
            month_idx=6, climate_idx=5, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        
        climate_hot_cool_ratio = np.median(climate_hot) / np.median(climate_cool)
        results["climate_zone_effect"] = {
            "hot_cool_ratio": climate_hot_cool_ratio,
            "expected_range": [1.5, 3.0],
            "passes": 1.5 <= climate_hot_cool_ratio <= 3.0
        }
        
        # 7. Pool effect
        no_pool = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        with_pool = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=1, rural=0, draws=draws
        )
        
        pool_effect_ratio = np.median(with_pool) / np.median(no_pool)
        results["pool_effect"] = {
            "pool_ratio": pool_effect_ratio,
            "expected_range": [3.0, 6.0],
            "passes": 2.5 <= pool_effect_ratio <= 8.0
        }
        
        # 8. Urban vs Rural effect
        urban_usage = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
        )
        rural_usage = self.predict_prior_draws(
            month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=1, draws=draws
        )
        
        rural_urban_ratio = np.median(rural_usage) / np.median(urban_usage)
        results["urban_rural_effect"] = {
            "rural_urban_ratio": rural_urban_ratio,
            "expected_ratio": 1.22,  # exp(0.2)
            "passes": 1.1 <= rural_urban_ratio <= 1.4
        }
        
        return results
    
    def predict_seasonal_usage(self, 
                              climate_idx: int,
                              hs: float, 
                              res_sf: int, 
                              lot: float, 
                              pool: int, 
                              rural: int,
                              draws: int = 2000) -> Dict[str, np.ndarray]:
        """
        Predict water usage across all 12 months for a specific scenario.
        
        Parameters
        ----------
        climate_idx : int
            Climate zone index (0 to K-1)
        hs : float
            Household size (persons)
        res_sf : int
            1 if single-family, 0 if multi-family
        lot : float
            Lot size in sqft
        pool : int
            1 if pool present, 0 otherwise
        rural : int
            1 if rural, 0 if urban
        draws : int
            Number of draws per month
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'months', 'median', 'q025', 'q975', 'q25', 'q75', 'mean'
        """
        if self.prior_samples is None:
            self.sample_priors()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_predictions = []
        
        for month_idx in range(12):
            month_draws = self.predict_prior_draws(
                month_idx=month_idx, climate_idx=climate_idx, hs=hs, 
                res_sf=res_sf, lot=lot, pool=pool, rural=rural, draws=draws
            )
            monthly_predictions.append(month_draws)
        
        # Stack all months into a single array (draws x 12)
        monthly_array = np.column_stack(monthly_predictions)
        
        # Calculate statistics for each month
        median = np.median(monthly_array, axis=0)
        q025 = np.percentile(monthly_array, 2.5, axis=0)
        q975 = np.percentile(monthly_array, 97.5, axis=0)
        q25 = np.percentile(monthly_array, 25, axis=0)
        q75 = np.percentile(monthly_array, 75, axis=0)
        mean = np.mean(monthly_array, axis=0)
        
        return {
            'months': month_names,
            'median': median,
            'q025': q025,
            'q975': q975,
            'q25': q25,
            'q75': q75,
            'mean': mean,
            'raw_data': monthly_array
        }
    
    def get_eugene_oregon_scenario(self) -> Dict:
        """
        Get scenario parameters for Eugene, Oregon.
        
        Returns
        -------
        Dict
            Scenario parameters for Eugene, Oregon
        """
        return {
            'climate_idx': 2,  # Moderate climate zone
            'hs': 4.0,         # Family of 4
            'res_sf': 1,       # Single-family home
            'lot': 8000,       # Typical lot size
            'pool': 0,         # No pool
            'rural': 1         # Rural area
        }
    
    def analyze_parameter_sensitivity(self, 
                                    base_scenario: Dict,
                                    parameter_ranges: Dict[str, List],
                                    draws: int = 1000) -> Dict[str, Dict]:
        """
        Analyze sensitivity of water usage to different parameters.
        
        Parameters
        ----------
        base_scenario : Dict
            Base scenario parameters
        parameter_ranges : Dict[str, List]
            Dictionary mapping parameter names to lists of values to test
        draws : int
            Number of draws per scenario
            
        Returns
        -------
        Dict[str, Dict]
            Sensitivity analysis results for each parameter
        """
        if self.prior_samples is None:
            self.sample_priors()
        
        results = {}
        
        for param_name, param_values in parameter_ranges.items():
            param_results = {
                'values': param_values,
                'medians': [],
                'q025': [],
                'q975': [],
                'q25': [],
                'q75': [],
                'means': []
            }
            
            for value in param_values:
                # Create scenario with modified parameter
                test_scenario = base_scenario.copy()
                test_scenario[param_name] = value
                
                # Get predictions for July (peak usage month)
                july_draws = self.predict_prior_draws(
                    month_idx=6, draws=draws, **test_scenario
                )
                
                param_results['medians'].append(np.median(july_draws))
                param_results['q025'].append(np.percentile(july_draws, 2.5))
                param_results['q975'].append(np.percentile(july_draws, 97.5))
                param_results['q25'].append(np.percentile(july_draws, 25))
                param_results['q75'].append(np.percentile(july_draws, 75))
                param_results['means'].append(np.mean(july_draws))
            
            results[param_name] = param_results
        
        return results
    
    def analyze_climate_impact(self, 
                              base_scenario: Dict,
                              draws: int = 1000) -> Dict[str, np.ndarray]:
        """
        Analyze water usage across all climate zones.
        
        Parameters
        ----------
        base_scenario : Dict
            Base scenario parameters
        draws : int
            Number of draws per climate zone
            
        Returns
        -------
        Dict[str, np.ndarray]
            Climate analysis results
        """
        if self.prior_samples is None:
            self.sample_priors()
        
        climate_results = {
            'zones': list(range(self.K)),
            'july_medians': [],
            'annual_medians': [],
            'july_q025': [],
            'july_q975': [],
            'annual_q025': [],
            'annual_q975': []
        }
        
        for climate_idx in range(self.K):
            test_scenario = base_scenario.copy()
            test_scenario['climate_idx'] = climate_idx
            
            # July usage (peak month)
            july_draws = self.predict_prior_draws(
                month_idx=6, draws=draws, **test_scenario
            )
            
            # Annual usage (sum across all months)
            annual_draws = []
            for month_idx in range(12):
                month_draws = self.predict_prior_draws(
                    month_idx=month_idx, draws=draws, **test_scenario
                )
                annual_draws.append(month_draws)
            
            annual_draws = np.sum(annual_draws, axis=0)
            
            climate_results['july_medians'].append(np.median(july_draws))
            climate_results['annual_medians'].append(np.median(annual_draws))
            climate_results['july_q025'].append(np.percentile(july_draws, 2.5))
            climate_results['july_q975'].append(np.percentile(july_draws, 97.5))
            climate_results['annual_q025'].append(np.percentile(annual_draws, 2.5))
            climate_results['annual_q975'].append(np.percentile(annual_draws, 97.5))
        
        return climate_results
    
    def analyze_household_size_impact(self, 
                                    base_scenario: Dict,
                                    household_sizes: List[float],
                                    draws: int = 1000) -> Dict[str, np.ndarray]:
        """
        Analyze water usage across different household sizes.
        
        Parameters
        ----------
        base_scenario : Dict
            Base scenario parameters
        household_sizes : List[float]
            List of household sizes to test
        draws : int
            Number of draws per household size
            
        Returns
        -------
        Dict[str, np.ndarray]
            Household size analysis results
        """
        if self.prior_samples is None:
            self.sample_priors()
        
        hs_results = {
            'sizes': household_sizes,
            'july_medians': [],
            'annual_medians': [],
            'july_q025': [],
            'july_q975': [],
            'annual_q025': [],
            'annual_q975': []
        }
        
        for hs in household_sizes:
            test_scenario = base_scenario.copy()
            test_scenario['hs'] = hs
            
            # July usage (peak month)
            july_draws = self.predict_prior_draws(
                month_idx=6, draws=draws, **test_scenario
            )
            
            # Annual usage (sum across all months)
            annual_draws = []
            for month_idx in range(12):
                month_draws = self.predict_prior_draws(
                    month_idx=month_idx, draws=draws, **test_scenario
                )
                annual_draws.append(month_draws)
            
            annual_draws = np.sum(annual_draws, axis=0)
            
            hs_results['july_medians'].append(np.median(july_draws))
            hs_results['annual_medians'].append(np.median(annual_draws))
            hs_results['july_q025'].append(np.percentile(july_draws, 2.5))
            hs_results['july_q975'].append(np.percentile(july_draws, 97.5))
            hs_results['annual_q025'].append(np.percentile(annual_draws, 2.5))
            hs_results['annual_q975'].append(np.percentile(annual_draws, 97.5))
        
        return hs_results
    
    def generate_scenario_matrix(self, 
                                parameter_grid: Dict[str, List],
                                draws: int = 500) -> pd.DataFrame:
        """
        Generate a matrix of scenarios with their predicted water usage.
        
        Parameters
        ----------
        parameter_grid : Dict[str, List]
            Dictionary mapping parameter names to lists of values
        draws : int
            Number of draws per scenario
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all scenario combinations and their predictions
        """
        if self.prior_samples is None:
            self.sample_priors()
        
        # Generate all combinations
        import itertools
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        
        for combo in combinations:
            scenario = dict(zip(param_names, combo))
            
            # Fill in missing parameters with defaults
            full_scenario = {
                'climate_idx': scenario.get('climate_idx', 2),
                'hs': scenario.get('hs', 4.0),
                'res_sf': scenario.get('res_sf', 1),
                'lot': scenario.get('lot', 8000),
                'pool': scenario.get('pool', 0),
                'rural': scenario.get('rural', 1)
            }
            
            # Get July usage (peak month)
            july_draws = self.predict_prior_draws(
                month_idx=6, draws=draws, **full_scenario
            )
            
            # Get annual usage
            annual_draws = []
            for month_idx in range(12):
                month_draws = self.predict_prior_draws(
                    month_idx=month_idx, draws=draws, **full_scenario
                )
                annual_draws.append(month_draws)
            
            annual_draws = np.sum(annual_draws, axis=0)
            
            result = scenario.copy()
            result['july_median'] = np.median(july_draws)
            result['july_q025'] = np.percentile(july_draws, 2.5)
            result['july_q975'] = np.percentile(july_draws, 97.5)
            result['annual_median'] = np.median(annual_draws)
            result['annual_q025'] = np.percentile(annual_draws, 2.5)
            result['annual_q975'] = np.percentile(annual_draws, 97.5)
            
            results.append(result)
        
        return pd.DataFrame(results)


def create_prior_visualization(model: WaterUsageModel, 
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create comprehensive visualization of prior distributions.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if model.prior_samples is None:
        model.sample_priors()
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot parameter distributions
    param_names = ["alpha", "beta_hs", "beta_res", "beta_lot", "beta_pool", 
                   "beta_rural", "sigma_y", "gamma", "eta"]
    
    for i, param in enumerate(param_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if param == "gamma":
            # Month effects
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            data = model.prior_samples[param]
            # Flatten the data for boxplot
            box_data = [data[:, j].flatten() for j in range(12)]
            ax.boxplot(box_data, tick_labels=month_names)
            ax.set_title("Month Effects (γ)")
            ax.set_ylabel("Log-scale effect")
            ax.tick_params(axis='x', rotation=45)
        elif param == "eta":
            # Climate zone effects
            data = model.prior_samples[param]
            # Flatten the data for boxplot
            box_data = [data[:, j].flatten() for j in range(model.K)]
            ax.boxplot(box_data)
            ax.set_title("Climate Zone Effects (η)")
            ax.set_ylabel("Log-scale effect")
            ax.set_xlabel("Climate Zone")
        else:
            # Other parameters
            data = model.prior_samples[param].flatten()
            ax.hist(data, bins=50, alpha=0.7, density=True)
            ax.set_title(f"{param}")
            ax.set_ylabel("Density")
            ax.set_xlabel("Value")
    
    plt.tight_layout()
    return fig


def create_prior_predictive_visualization(model: WaterUsageModel, 
                                        draws: int = 2000) -> plt.Figure:
    """
    Create visualization of prior predictive checks.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    draws : int
        Number of draws per scenario
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    scenarios = model.get_prior_scenarios()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, sc) in enumerate(scenarios.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        draws_array = model.predict_prior_draws(**sc, draws=draws)
        
        # Create histogram
        ax.hist(draws_array, bins=50, alpha=0.7, density=True)
        ax.axvline(np.median(draws_array), color='red', linestyle='--', 
                  label=f'Median: {np.median(draws_array):.0f}')
        ax.axvline(np.mean(draws_array), color='orange', linestyle='--', 
                  label=f'Mean: {np.mean(draws_array):.0f}')
        
        ax.set_title(f"{name.replace('_', ' ').title()}")
        ax.set_xlabel("Monthly Water Usage (gallons)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Overall distribution
    ax = axes[-1]
    overall_draws = model.predict_prior_draws(
        month_idx=0, climate_idx=2, hs=model.hs_mean, 
        res_sf=1, lot=model.lot_mean, pool=0, rural=0, draws=5000
    )
    ax.hist(overall_draws, bins=50, alpha=0.7, density=True)
    ax.axvline(np.median(overall_draws), color='red', linestyle='--', 
              label=f'Median: {np.median(overall_draws):.0f}')
    ax.set_title("Overall Prior Predictive\n(Average Household, January)")
    ax.set_xlabel("Monthly Water Usage (gallons)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_research_validation_visualization(model: WaterUsageModel, 
                                           draws: int = 2000) -> plt.Figure:
    """
    Create comprehensive visualization of research validation checks.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    draws : int
        Number of draws per scenario
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Run comprehensive checks
    results = model.run_comprehensive_prior_predictive_checks(draws)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # 1. Baseline usage comparison (Winter vs Summer)
    ax = axes[0]
    winter_draws = model.predict_prior_draws(
        month_idx=0, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    summer_draws = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    
    ax.hist(winter_draws, bins=50, alpha=0.6, label='Winter', color='blue', density=True)
    ax.hist(summer_draws, bins=50, alpha=0.6, label='Summer', color='red', density=True)
    ax.axvline(np.median(winter_draws), color='blue', linestyle='--', alpha=0.8)
    ax.axvline(np.median(summer_draws), color='red', linestyle='--', alpha=0.8)
    ax.set_title('Baseline Usage: Winter vs Summer')
    ax.set_xlabel('Monthly Water Usage (gallons)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Household size effect
    ax = axes[1]
    hs_3 = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    hs_4 = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=4.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    hs_5 = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=5.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    
    ax.hist(hs_3, bins=50, alpha=0.6, label='3 people', color='green', density=True)
    ax.hist(hs_4, bins=50, alpha=0.6, label='4 people', color='orange', density=True)
    ax.hist(hs_5, bins=50, alpha=0.6, label='5 people', color='purple', density=True)
    ax.set_title('Household Size Effect')
    ax.set_xlabel('Monthly Water Usage (gallons)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. SF vs MF comparison
    ax = axes[2]
    sf_usage = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    mf_usage = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=0, lot=8000, pool=0, rural=0, draws=draws
    )
    
    ax.hist(sf_usage, bins=50, alpha=0.6, label='Single-Family', color='brown', density=True)
    ax.hist(mf_usage, bins=50, alpha=0.6, label='Multi-Family', color='pink', density=True)
    ax.axvline(np.median(sf_usage), color='brown', linestyle='--', alpha=0.8)
    ax.axvline(np.median(mf_usage), color='pink', linestyle='--', alpha=0.8)
    ax.set_title('SF vs MF Comparison')
    ax.set_xlabel('Monthly Water Usage (gallons)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Lot size effect
    ax = axes[3]
    lot_small = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=2000, pool=0, rural=0, draws=draws
    )
    lot_medium = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    lot_large = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=14000, pool=0, rural=0, draws=draws
    )
    
    ax.hist(lot_small, bins=50, alpha=0.6, label='Small lot (2k sqft)', color='lightblue', density=True)
    ax.hist(lot_medium, bins=50, alpha=0.6, label='Medium lot (8k sqft)', color='blue', density=True)
    ax.hist(lot_large, bins=50, alpha=0.6, label='Large lot (14k sqft)', color='darkblue', density=True)
    ax.set_title('Lot Size Effect')
    ax.set_xlabel('Monthly Water Usage (gallons)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Climate zone effect
    ax = axes[4]
    climate_cool = model.predict_prior_draws(
        month_idx=6, climate_idx=0, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    climate_moderate = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    climate_hot = model.predict_prior_draws(
        month_idx=6, climate_idx=5, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    
    ax.hist(climate_cool, bins=50, alpha=0.6, label='Cool climate', color='blue', density=True)
    ax.hist(climate_moderate, bins=50, alpha=0.6, label='Moderate climate', color='green', density=True)
    ax.hist(climate_hot, bins=50, alpha=0.6, label='Hot climate', color='red', density=True)
    ax.set_title('Climate Zone Effect')
    ax.set_xlabel('Monthly Water Usage (gallons)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Pool effect
    ax = axes[5]
    no_pool = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    with_pool = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=1, rural=0, draws=draws
    )
    
    ax.hist(no_pool, bins=50, alpha=0.6, label='No pool', color='lightgreen', density=True)
    ax.hist(with_pool, bins=50, alpha=0.6, label='With pool', color='darkgreen', density=True)
    ax.axvline(np.median(no_pool), color='lightgreen', linestyle='--', alpha=0.8)
    ax.axvline(np.median(with_pool), color='darkgreen', linestyle='--', alpha=0.8)
    ax.set_title('Pool Effect')
    ax.set_xlabel('Monthly Water Usage (gallons)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Urban vs Rural effect
    ax = axes[6]
    urban_usage = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=0, draws=draws
    )
    rural_usage = model.predict_prior_draws(
        month_idx=6, climate_idx=2, hs=3.0, res_sf=1, lot=8000, pool=0, rural=1, draws=draws
    )
    
    ax.hist(urban_usage, bins=50, alpha=0.6, label='Urban', color='gray', density=True)
    ax.hist(rural_usage, bins=50, alpha=0.6, label='Rural', color='brown', density=True)
    ax.axvline(np.median(urban_usage), color='gray', linestyle='--', alpha=0.8)
    ax.axvline(np.median(rural_usage), color='brown', linestyle='--', alpha=0.8)
    ax.set_title('Urban vs Rural Effect')
    ax.set_xlabel('Monthly Water Usage (gallons)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Validation summary
    ax = axes[7]
    ax.axis('off')
    
    # Create validation summary text
    summary_text = "VALIDATION SUMMARY\n" + "="*30 + "\n\n"
    
    checks = [
        ("Seasonal Pattern", results["seasonal_pattern"]["passes"]),
        ("Household Size (4v3)", results["household_size_effect"]["passes_4_3"]),
        ("Household Size (5v3)", results["household_size_effect"]["passes_5_3"]),
        ("SF vs MF", results["sf_vs_mf"]["passes"]),
        ("Lot Size (1σ)", results["lot_size_effect"]["passes_1std"]),
        ("Lot Size (2σ)", results["lot_size_effect"]["passes_2std"]),
        ("Climate Zones", results["climate_zone_effect"]["passes"]),
        ("Pool Effect", results["pool_effect"]["passes"]),
        ("Urban/Rural", results["urban_rural_effect"]["passes"])
    ]
    
    for check_name, passes in checks:
        status = "✅ PASS" if passes else "❌ FAIL"
        summary_text += f"{check_name:<20} {status}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def create_seasonal_usage_plot(model: WaterUsageModel,
                              scenario_params: Dict,
                              draws: int = 2000,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a seasonal water usage plot with 95% confidence intervals.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    scenario_params : Dict
        Scenario parameters (climate_idx, hs, res_sf, lot, pool, rural)
    draws : int
        Number of draws per month
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Get seasonal predictions
    seasonal_data = model.predict_seasonal_usage(draws=draws, **scenario_params)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    months = seasonal_data['months']
    median = seasonal_data['median']
    q025 = seasonal_data['q025']
    q975 = seasonal_data['q975']
    q25 = seasonal_data['q25']
    q75 = seasonal_data['q75']
    
    # Create x-axis positions
    x_pos = np.arange(len(months))
    
    # Plot 95% confidence interval
    ax.fill_between(x_pos, q025, q975, alpha=0.2, color='blue', 
                   label='95% Confidence Interval')
    
    # Plot 50% confidence interval (IQR)
    ax.fill_between(x_pos, q25, q75, alpha=0.3, color='blue', 
                   label='50% Confidence Interval (IQR)')
    
    # Plot median line
    ax.plot(x_pos, median, 'o-', color='darkblue', linewidth=2, markersize=6,
           label='Median Prediction')
    
    # Customize the plot
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Monthly Water Usage (gallons)', fontsize=12)
    ax.set_title('Seasonal Water Usage Prediction', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(months, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add scenario description
    scenario_desc = f"Climate Zone: {scenario_params['climate_idx']}, "
    scenario_desc += f"Household Size: {scenario_params['hs']}, "
    scenario_desc += f"Residence: {'Single-Family' if scenario_params['res_sf'] else 'Multi-Family'}, "
    scenario_desc += f"Lot Size: {scenario_params['lot']:,} sqft, "
    scenario_desc += f"Pool: {'Yes' if scenario_params['pool'] else 'No'}, "
    scenario_desc += f"Location: {'Rural' if scenario_params['rural'] else 'Urban'}"
    
    ax.text(0.02, 0.98, scenario_desc, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.8))
    
    # Add summary statistics
    annual_median = np.sum(median)
    peak_month = months[np.argmax(median)]
    peak_usage = np.max(median)
    winter_usage = np.mean(median[0:2])  # Jan-Feb average
    summer_usage = np.mean(median[6:8])  # Jul-Aug average
    seasonal_ratio = summer_usage / winter_usage
    
    stats_text = f"Annual Total: {annual_median:,.0f} gallons\n"
    stats_text += f"Peak Month: {peak_month} ({peak_usage:,.0f} gal)\n"
    stats_text += f"Summer/Winter Ratio: {seasonal_ratio:.1f}x"
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_sensitivity_analysis_plot(model: WaterUsageModel,
                                   base_scenario: Dict,
                                   parameter_ranges: Dict[str, List],
                                   draws: int = 1000,
                                   figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create sensitivity analysis plots for multiple parameters.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    base_scenario : Dict
        Base scenario parameters
    parameter_ranges : Dict[str, List]
        Dictionary mapping parameter names to lists of values to test
    draws : int
        Number of draws per scenario
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Run sensitivity analysis
    sensitivity_results = model.analyze_parameter_sensitivity(
        base_scenario, parameter_ranges, draws
    )
    
    n_params = len(parameter_ranges)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (param_name, results) in enumerate(sensitivity_results.items()):
        ax = axes[i]
        
        values = results['values']
        medians = results['medians']
        q025 = results['q025']
        q975 = results['q975']
        q25 = results['q25']
        q75 = results['q75']
        
        # Plot confidence intervals
        ax.fill_between(values, q025, q975, alpha=0.2, color='blue', 
                       label='95% CI')
        ax.fill_between(values, q25, q75, alpha=0.3, color='blue', 
                       label='50% CI')
        ax.plot(values, medians, 'o-', color='darkblue', linewidth=2, 
               markersize=6, label='Median')
        
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('July Water Usage (gallons)')
        ax.set_title(f'Sensitivity to {param_name.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def create_climate_impact_plot(model: WaterUsageModel,
                              base_scenario: Dict,
                              draws: int = 1000,
                              figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create climate impact analysis plot.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    base_scenario : Dict
        Base scenario parameters
    draws : int
        Number of draws per climate zone
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Run climate analysis
    climate_results = model.analyze_climate_impact(base_scenario, draws)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    zones = climate_results['zones']
    july_medians = climate_results['july_medians']
    annual_medians = climate_results['annual_medians']
    july_q025 = climate_results['july_q025']
    july_q975 = climate_results['july_q975']
    annual_q025 = climate_results['annual_q025']
    annual_q975 = climate_results['annual_q975']
    
    # July usage by climate zone
    ax1.fill_between(zones, july_q025, july_q975, alpha=0.2, color='red')
    ax1.plot(zones, july_medians, 'o-', color='darkred', linewidth=2, markersize=6)
    ax1.set_xlabel('Climate Zone')
    ax1.set_ylabel('July Water Usage (gallons)')
    ax1.set_title('Peak Month Usage by Climate Zone')
    ax1.grid(True, alpha=0.3)
    
    # Annual usage by climate zone
    ax2.fill_between(zones, annual_q025, annual_q975, alpha=0.2, color='blue')
    ax2.plot(zones, annual_medians, 'o-', color='darkblue', linewidth=2, markersize=6)
    ax2.set_xlabel('Climate Zone')
    ax2.set_ylabel('Annual Water Usage (gallons)')
    ax2.set_title('Annual Usage by Climate Zone')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Climate Zone Impact Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def create_household_size_analysis_plot(model: WaterUsageModel,
                                       base_scenario: Dict,
                                       household_sizes: List[float],
                                       draws: int = 1000,
                                       figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create household size impact analysis plot.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    base_scenario : Dict
        Base scenario parameters
    household_sizes : List[float]
        List of household sizes to test
    draws : int
        Number of draws per household size
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Run household size analysis
    hs_results = model.analyze_household_size_impact(base_scenario, household_sizes, draws)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    sizes = hs_results['sizes']
    july_medians = hs_results['july_medians']
    annual_medians = hs_results['annual_medians']
    july_q025 = hs_results['july_q025']
    july_q975 = hs_results['july_q975']
    annual_q025 = hs_results['annual_q025']
    annual_q975 = hs_results['annual_q975']
    
    # July usage by household size
    ax1.fill_between(sizes, july_q025, july_q975, alpha=0.2, color='green')
    ax1.plot(sizes, july_medians, 'o-', color='darkgreen', linewidth=2, markersize=6)
    ax1.set_xlabel('Household Size (people)')
    ax1.set_ylabel('July Water Usage (gallons)')
    ax1.set_title('Peak Month Usage by Household Size')
    ax1.grid(True, alpha=0.3)
    
    # Annual usage by household size
    ax2.fill_between(sizes, annual_q025, annual_q975, alpha=0.2, color='purple')
    ax2.plot(sizes, annual_medians, 'o-', color='darkviolet', linewidth=2, markersize=6)
    ax2.set_xlabel('Household Size (people)')
    ax2.set_ylabel('Annual Water Usage (gallons)')
    ax2.set_title('Annual Usage by Household Size')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Household Size Impact Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def create_scenario_heatmap(model: WaterUsageModel,
                           parameter_grid: Dict[str, List],
                           metric: str = 'annual_median',
                           draws: int = 500,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a heatmap of scenario predictions.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    parameter_grid : Dict[str, List]
        Dictionary mapping parameter names to lists of values
    metric : str
        Metric to plot ('annual_median', 'july_median', etc.)
    draws : int
        Number of draws per scenario
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Generate scenario matrix
    scenario_df = model.generate_scenario_matrix(parameter_grid, draws)
    
    # For heatmap, we need exactly 2 parameters
    param_names = list(parameter_grid.keys())
    if len(param_names) != 2:
        raise ValueError("Heatmap requires exactly 2 parameters")
    
    param1, param2 = param_names
    values1 = parameter_grid[param1]
    values2 = parameter_grid[param2]
    
    # Create pivot table for heatmap
    heatmap_data = scenario_df.pivot_table(
        values=metric, 
        index=param2, 
        columns=param1, 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(values1)))
    ax.set_yticks(range(len(values2)))
    ax.set_xticklabels(values1)
    ax.set_yticklabels(values2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric.replace("_", " ").title()} (gallons)')
    
    # Add value annotations
    for i in range(len(values2)):
        for j in range(len(values1)):
            text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xlabel(param1.replace('_', ' ').title())
    ax.set_ylabel(param2.replace('_', ' ').title())
    ax.set_title(f'Scenario Heatmap: {metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    return fig


def create_distribution_comparison_plot(model: WaterUsageModel,
                                      scenarios: Dict[str, Dict],
                                      month_idx: int = 6,
                                      draws: int = 2000,
                                      figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Create distribution comparison plots for multiple scenarios.
    
    Parameters
    ----------
    model : WaterUsageModel
        The water usage model instance
    scenarios : Dict[str, Dict]
        Dictionary of scenario names and parameters
    month_idx : int
        Month to analyze (0-11)
    draws : int
        Number of draws per scenario
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    n_scenarios = len(scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_scenarios == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, (scenario_name, scenario_params) in enumerate(scenarios.items()):
        ax = axes[i]
        
        # Get predictions for this scenario
        draws_array = model.predict_prior_draws(
            month_idx=month_idx, draws=draws, **scenario_params
        )
        
        # Create histogram
        ax.hist(draws_array, bins=50, alpha=0.7, density=True, color='skyblue')
        ax.axvline(np.median(draws_array), color='red', linestyle='--', 
                  label=f'Median: {np.median(draws_array):.0f}')
        ax.axvline(np.mean(draws_array), color='orange', linestyle='--', 
                  label=f'Mean: {np.mean(draws_array):.0f}')
        
        # Add percentiles
        q025 = np.percentile(draws_array, 2.5)
        q975 = np.percentile(draws_array, 97.5)
        ax.axvline(q025, color='gray', linestyle=':', alpha=0.7, 
                  label=f'2.5%: {q025:.0f}')
        ax.axvline(q975, color='gray', linestyle=':', alpha=0.7, 
                  label=f'97.5%: {q975:.0f}')
        
        ax.set_title(f'{scenario_name}\n{month_names[month_idx]} Usage')
        ax.set_xlabel('Monthly Water Usage (gallons)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_scenarios, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Distribution Comparison: {month_names[month_idx]} Usage', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    return fig
