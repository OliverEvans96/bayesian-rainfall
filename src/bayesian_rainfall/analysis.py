"""
Analysis functions for Bayesian rainfall model results.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import pymc as pm
import matplotlib.dates as mdates


def _parse_date_input(date_input):
    """
    Parse date input that can be either day_of_year (int) or month/day (str or tuple).
    
    Parameters:
    -----------
    date_input : int, str, or tuple
        - int: day of year (1-365)
        - str: "MM/DD" format (e.g., "01/15")
        - tuple: (month, day) format (e.g., (1, 15))
    
    Returns:
    --------
    tuple : (day_of_year, day_name)
    """
    if isinstance(date_input, int):
        day_of_year = date_input
        if day_of_year < 1 or day_of_year > 365:
            raise ValueError("day_of_year must be between 1 and 365")
        
        # Convert day of year to month/day
        date_obj = datetime(2024, 1, 1) + pd.Timedelta(days=day_of_year - 1)
        day_name = date_obj.strftime("%B %d")
        
    elif isinstance(date_input, str):
        # Parse "MM/DD" format
        try:
            month, day = map(int, date_input.split('/'))
            if month < 1 or month > 12 or day < 1 or day > 31:
                raise ValueError("Invalid month or day")
            
            # Convert to day of year
            date_obj = datetime(2024, month, day)
            day_of_year = date_obj.timetuple().tm_yday
            day_name = date_obj.strftime("%B %d")
            
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_input}'. Use 'MM/DD' format (e.g., '01/15')") from e
            
    elif isinstance(date_input, (tuple, list)) and len(date_input) == 2:
        # Parse (month, day) format
        month, day = date_input
        if not (1 <= month <= 12) or not (1 <= day <= 31):
            raise ValueError("Month must be 1-12, day must be 1-31")
        
        # Convert to day of year
        date_obj = datetime(2024, month, day)
        day_of_year = date_obj.timetuple().tm_yday
        day_name = date_obj.strftime("%B %d")
        
    else:
        raise ValueError("date_input must be int (day_of_year), str ('MM/DD'), or tuple (month, day)")
    
    return day_of_year, day_name


def _evaluate_model_for_day(trace, day_of_year, year=None):
    """
    Helper function to evaluate hierarchical model for a specific day using posterior samples.

    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from hierarchical model sampling
    day_of_year : int
        Day of year (1-365)
    year : int, optional
        Year for year-specific prediction. If None, uses average year effects.

    Returns:
    --------
    tuple : (rain_probs, expected_amounts, alpha_amounts)
        Arrays of rain probabilities, expected rainfall amounts, and alpha parameters
    """
    # Get the number of harmonics from the trace
    n_harmonics = trace.posterior.a_rain.shape[-1]
    
    # Create harmonic features for the specific day (vectorized)
    h_values = np.arange(1, n_harmonics + 1)  # Shape: (n_harmonics,)
    day_sin = np.sin(2 * h_values * np.pi * day_of_year / 365.25)  # Shape: (n_harmonics,)
    day_cos = np.cos(2 * h_values * np.pi * day_of_year / 365.25)  # Shape: (n_harmonics,)
    
    # Get parameter samples
    a_rain_samples = trace.posterior.a_rain.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    b_rain_samples = trace.posterior.b_rain.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    c_rain_samples = trace.posterior.c_rain.values.flatten()  # Shape: (n_samples,)
    a_amount_samples = trace.posterior.a_amount.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    b_amount_samples = trace.posterior.b_amount.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    c_amount_samples = trace.posterior.c_amount.values.flatten()  # Shape: (n_samples,)
    alpha_amount_samples = trace.posterior.alpha_amount.values.flatten()  # Shape: (n_samples,)
    
    # Get year effects
    year_rain_effects = trace.posterior.year_rain_effects.values  # Shape: (chains, draws, n_years)
    year_amount_effects = trace.posterior.year_amount_effects.values
    
    # Get unique years from the model
    unique_years = trace.posterior.year_rain_effects.coords['year'].values
    
    # Handle year selection
    if year is not None:
        if year not in unique_years:
            raise ValueError(f"Year {year} not found in model. Available years: {unique_years}")
        year_idx = np.where(unique_years == year)[0][0]
        year_rain_effect = year_rain_effects[:, :, year_idx].flatten()  # Shape: (n_samples,)
        year_amount_effect = year_amount_effects[:, :, year_idx].flatten()  # Shape: (n_samples,)
    else:
        # Use average year effects (all zeros since they're centered)
        year_rain_effect = np.zeros_like(c_rain_samples)
        year_amount_effect = np.zeros_like(c_amount_samples)
    
    # Calculate rain probabilities for all samples (vectorized)
    # a_rain_samples: (n_samples, n_harmonics), day_sin: (n_harmonics,)
    # Result: (n_samples,) - sum over harmonics dimension
    logit_p = c_rain_samples + np.sum(a_rain_samples * day_sin + b_rain_samples * day_cos, axis=1) + year_rain_effect
    rain_probs = 1 / (1 + np.exp(-logit_p))
    
    # Calculate expected rainfall amounts (vectorized)
    # a_amount_samples: (n_samples, n_harmonics), day_sin: (n_harmonics,)
    # Result: (n_samples,) - sum over harmonics dimension
    log_mu_amount = c_amount_samples + np.sum(a_amount_samples * day_sin + b_amount_samples * day_cos, axis=1) + year_amount_effect
    expected_amounts = np.exp(log_mu_amount)
    
    return rain_probs, expected_amounts, alpha_amount_samples


def sample_posterior_predictive_for_day(trace, day_of_year, n_samples=1000):
    """
    Sample posterior predictive for a specific day using direct evaluation.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    day_of_year : int
        Day of year (1-365)
    n_samples : int
        Number of posterior predictive samples to generate
    
    Returns:
    --------
    tuple : (rain_indicators, rainfall_amounts)
        Arrays of rain indicators and rainfall amounts
    """
    # Use helper function to get model predictions
    rain_probs, expected_amounts, alpha_amounts = _evaluate_model_for_day(trace, day_of_year)
    
    # Limit to requested number of samples
    n_samples = min(n_samples, len(rain_probs))
    rain_probs = rain_probs[:n_samples]
    expected_amounts = expected_amounts[:n_samples]
    alpha_amounts = alpha_amounts[:n_samples]
    
    # Generate rainfall samples
    rain_indicators = []
    rainfall_amounts = []
    
    for i, (p_rain, mu_amount, alpha_amount) in enumerate(zip(rain_probs, expected_amounts, alpha_amounts)):
        # Sample rain indicator
        rain_indicator = np.random.binomial(1, p_rain)
        rain_indicators.append(rain_indicator)
        
        # Sample rainfall amount if it rains
        if rain_indicator == 1:
            rainfall = np.random.gamma(alpha_amount, mu_amount / alpha_amount)
        else:
            rainfall = 0
        rainfall_amounts.append(rainfall)
    
    return np.array(rain_indicators), np.array(rainfall_amounts)


def print_model_summary(trace, data, param_names=None):
    """
    Print a comprehensive summary of the model results including statistics,
    convergence diagnostics, seasonal patterns, and model fit analysis.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    data : pandas.DataFrame
        Weather data with columns 'PRCP' and 'day_of_year'
    param_names : list, optional
        List of parameter names to check convergence. If None, uses main parameters.
    """
    if param_names is None:
        param_names = ['a_rain', 'b_rain', 'c_rain', 'a_amount', 'b_amount', 'c_amount', 'alpha_amount']
    
    # Get the number of harmonics from the trace
    n_harmonics = trace.posterior.a_rain.shape[-1]
    
    # Calculate seasonal patterns for rain probability and amounts (vectorized)
    days_of_year = np.arange(1, 366)
    
    # Create harmonic features for all days (vectorized)
    h_values = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
    day_values = days_of_year[None, :]  # Shape: (1, n_days)
    
    # Vectorized harmonic calculations
    day_sin_pred = np.sin(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
    day_cos_pred = np.cos(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
    
    # Get posterior samples
    a_rain_samples = trace.posterior.a_rain.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    b_rain_samples = trace.posterior.b_rain.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    c_rain_samples = trace.posterior.c_rain.values.flatten()  # Shape: (n_samples,)
    a_amount_samples = trace.posterior.a_amount.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    b_amount_samples = trace.posterior.b_amount.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    c_amount_samples = trace.posterior.c_amount.values.flatten()  # Shape: (n_samples,)
    
    # Calculate rain probabilities (vectorized)
    # a_rain_samples: (n_samples, n_harmonics), day_sin_pred: (n_harmonics, n_days)
    # Result: (n_samples, n_days) - sum over harmonics dimension
    logit_p = c_rain_samples[:, None] + np.sum(a_rain_samples[:, :, None] * day_sin_pred[None, :, :] + 
                                               b_rain_samples[:, :, None] * day_cos_pred[None, :, :], axis=1)
    rain_probs = 1 / (1 + np.exp(-logit_p))  # Shape: (n_samples, n_days)
    rain_prob_mean = np.mean(rain_probs, axis=0)  # Shape: (n_days,)
    
    # Calculate rainfall amounts (vectorized)
    # a_amount_samples: (n_samples, n_harmonics), day_sin_pred: (n_harmonics, n_days)
    # Result: (n_samples, n_days) - sum over harmonics dimension
    log_mu_amount = c_amount_samples[:, None] + np.sum(a_amount_samples[:, :, None] * day_sin_pred[None, :, :] + 
                                                      b_amount_samples[:, :, None] * day_cos_pred[None, :, :], axis=1)
    rainfall_amounts = np.exp(log_mu_amount)  # Shape: (n_samples, n_days)
    rainfall_mean = np.mean(rainfall_amounts, axis=0)  # Shape: (n_days,)
    
    print("=== MODEL SUMMARY ===")
    print(f"Total observations: {len(data)}")
    print(f"Rainy days: {len(data[data['PRCP'] > 0])}")
    print(f"Overall rain frequency: {len(data[data['PRCP'] > 0]) / len(data):.3f}")
    print(f"Mean rainfall on rainy days: {data[data['PRCP'] > 0]['PRCP'].mean():.3f} mm")
    print(f"Max rainfall: {data['PRCP'].max():.3f} mm")

    print("\n=== CONVERGENCE DIAGNOSTICS ===")
    print("R-hat values (should be < 1.01):")
    for param in param_names:
        rhat = az.rhat(trace, var_names=[param])[param].values
        # Handle different array shapes
        if np.isscalar(rhat):
            print(f"  {param}: {rhat:.4f}")
        elif rhat.size == 1:
            print(f"  {param}: {rhat.item():.4f}")
        else:
            # For multi-dimensional parameters, show mean and range
            rhat_flat = rhat.flatten()
            print(f"  {param}: mean={rhat_flat.mean():.4f}, range=[{rhat_flat.min():.4f}, {rhat_flat.max():.4f}]")

    print("\n=== MODEL FIT ANALYSIS ===")
    # Calculate model predictions for observed days (vectorized)
    observed_days = data['day_of_year'].values
    
    # Create harmonic features for observed days (vectorized)
    h_values_obs = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
    day_values_obs = observed_days[None, :]  # Shape: (1, n_obs_days)
    
    # Vectorized harmonic calculations for observed days
    day_sin_obs = np.sin(2 * h_values_obs * np.pi * day_values_obs / 365.25)  # Shape: (n_harmonics, n_obs_days)
    day_cos_obs = np.cos(2 * h_values_obs * np.pi * day_values_obs / 365.25)  # Shape: (n_harmonics, n_obs_days)
    
    # Calculate predicted rain probabilities for observed days (vectorized)
    # a_rain_samples: (n_samples, n_harmonics), day_sin_obs: (n_harmonics, n_obs_days)
    # Result: (n_samples, n_obs_days) - sum over harmonics dimension
    logit_p_obs = c_rain_samples[:, None] + np.sum(a_rain_samples[:, :, None] * day_sin_obs[None, :, :] + 
                                                   b_rain_samples[:, :, None] * day_cos_obs[None, :, :], axis=1)
    predicted_rain_probs = 1 / (1 + np.exp(-logit_p_obs))  # Shape: (n_samples, n_obs_days)
    predicted_rain_prob_mean = np.mean(predicted_rain_probs, axis=0)  # Shape: (n_obs_days,)
    
    # Calculate predicted rainfall amounts for observed days (vectorized)
    # a_amount_samples: (n_samples, n_harmonics), day_sin_obs: (n_harmonics, n_obs_days)
    # Result: (n_samples, n_obs_days) - sum over harmonics dimension
    log_mu_amount_obs = c_amount_samples[:, None] + np.sum(a_amount_samples[:, :, None] * day_sin_obs[None, :, :] + 
                                                          b_amount_samples[:, :, None] * day_cos_obs[None, :, :], axis=1)
    predicted_rainfall_amounts = np.exp(log_mu_amount_obs)  # Shape: (n_samples, n_obs_days)
    predicted_rainfall_mean = np.mean(predicted_rainfall_amounts, axis=0)  # Shape: (n_obs_days,)
    
    # Rain frequency comparison
    observed_rain_freq = (data['PRCP'] > 0).mean()
    predicted_rain_freq = predicted_rain_prob_mean.mean()
    rain_freq_error = abs(observed_rain_freq - predicted_rain_freq)
    rain_freq_error_pct = (rain_freq_error / observed_rain_freq) * 100
    
    print(f"Rain frequency - Observed: {observed_rain_freq:.3f}, Predicted: {predicted_rain_freq:.3f}")
    print(f"Rain frequency error: {rain_freq_error:.3f} ({rain_freq_error_pct:.1f}%)")
    
    # Rainfall amount comparison (only for rainy days)
    rainy_data = data[data['PRCP'] > 0]
    if len(rainy_data) > 0:
        observed_rainfall_mean = rainy_data['PRCP'].mean()
        # Get predicted amounts only for days when it actually rained
        rainy_mask = data['PRCP'] > 0
        predicted_rainy_amounts = predicted_rainfall_mean[rainy_mask]
        predicted_rainfall_mean_rainy = predicted_rainy_amounts.mean()
        
        rainfall_error = abs(observed_rainfall_mean - predicted_rainfall_mean_rainy)
        rainfall_error_pct = (rainfall_error / observed_rainfall_mean) * 100
        
        print(f"Rainfall amount (rainy days) - Observed: {observed_rainfall_mean:.3f}, Predicted: {predicted_rainfall_mean_rainy:.3f}")
        print(f"Rainfall amount error: {rainfall_error:.3f} ({rainfall_error_pct:.1f}%)")
    
    # Seasonal fit analysis
    data['month'] = pd.to_datetime(data['DATE']).dt.month
    monthly_obs_rain_freq = data.groupby('month')['PRCP'].apply(lambda x: (x > 0).mean())
    monthly_pred_rain_freq = []
    
    for month in range(1, 13):
        month_data = data[data['month'] == month]
        if len(month_data) > 0:
            month_days = month_data['day_of_year'].values
            
            # Create harmonic features for month days (vectorized)
            h_values_month = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
            day_values_month = month_days[None, :]  # Shape: (1, n_month_days)
            
            # Vectorized harmonic calculations for month days
            month_sin = np.sin(2 * h_values_month * np.pi * day_values_month / 365.25)  # Shape: (n_harmonics, n_month_days)
            month_cos = np.cos(2 * h_values_month * np.pi * day_values_month / 365.25)  # Shape: (n_harmonics, n_month_days)
            
            # Calculate month rain probabilities (vectorized)
            # a_rain_samples: (n_samples, n_harmonics), month_sin: (n_harmonics, n_month_days)
            # Result: (n_samples, n_month_days) - sum over harmonics dimension
            logit_p_month = c_rain_samples[:, None] + np.sum(a_rain_samples[:, :, None] * month_sin[None, :, :] + 
                                                             b_rain_samples[:, :, None] * month_cos[None, :, :], axis=1)
            month_rain_probs = 1 / (1 + np.exp(-logit_p_month))  # Shape: (n_samples, n_month_days)
            monthly_pred_rain_freq.append(np.mean(month_rain_probs))
        else:
            monthly_pred_rain_freq.append(0)
    
    monthly_pred_rain_freq = np.array(monthly_pred_rain_freq)
    monthly_mae = np.mean(np.abs(monthly_obs_rain_freq.values - monthly_pred_rain_freq))
    
    print(f"Monthly rain frequency MAE: {monthly_mae:.3f}")
    
    # Model uncertainty analysis
    rain_prob_std = np.std(predicted_rain_probs, axis=0)
    avg_uncertainty = np.mean(rain_prob_std)
    print(f"Average prediction uncertainty (std): {avg_uncertainty:.3f}")
    
    # Overall model assessment
    if rain_freq_error_pct < 5 and rainfall_error_pct < 10 and monthly_mae < 0.1:
        fit_quality = "Excellent"
    elif rain_freq_error_pct < 10 and rainfall_error_pct < 20 and monthly_mae < 0.15:
        fit_quality = "Good"
    elif rain_freq_error_pct < 20 and rainfall_error_pct < 30 and monthly_mae < 0.25:
        fit_quality = "Fair"
    else:
        fit_quality = "Poor"
    
    print(f"\nOverall model fit quality: {fit_quality}")

    print("\n=== SEASONAL PATTERNS ===")
    # Find peak and minimum rain probability days
    peak_day = days_of_year[np.argmax(rain_prob_mean)]
    min_day = days_of_year[np.argmin(rain_prob_mean)]
    print(f"Peak rain probability: Day {peak_day} ({rain_prob_mean[np.argmax(rain_prob_mean)]:.3f})")
    print(f"Minimum rain probability: Day {min_day} ({rain_prob_mean[np.argmin(rain_prob_mean)]:.3f})")

    # Find peak and minimum rainfall amounts
    peak_amount_day = days_of_year[np.argmax(rainfall_mean)]
    min_amount_day = days_of_year[np.argmin(rainfall_mean)]
    print(f"Peak rainfall amount: Day {peak_amount_day} ({rainfall_mean[np.argmax(rainfall_mean)]:.3f} mm)")
    print(f"Minimum rainfall amount: Day {min_amount_day} ({rainfall_mean[np.argmin(rainfall_mean)]:.3f} mm)")


def print_convergence_diagnostics(trace, param_names=None):
    """
    Print convergence diagnostics for the MCMC sampling.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    param_names : list, optional
        List of parameter names to check convergence. If None, uses main parameters.
    """
    if param_names is None:
        param_names = ['a_rain', 'b_rain', 'c_rain', 'a_amount', 'b_amount', 'c_amount', 'alpha_amount']
    
    print("R-hat values (should be < 1.01 for good convergence):")
    print(az.rhat(trace))
    print("\nEffective sample size:")
    print(az.ess(trace))


def analyze_single_day(trace, data, date_input, show_plots=True, figsize=(15, 10)):
    """
    Comprehensive analysis of model predictions for a single day-of-year.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    data : pandas.DataFrame
        Weather data with columns 'PRCP' and 'day_of_year'
    date_input : int, str, or tuple
        - int: day of year (1-365)
        - str: "MM/DD" format (e.g., "01/15")
        - tuple: (month, day) format (e.g., (1, 15))
    show_plots : bool, optional
        Whether to display plots (default True)
    figsize : tuple, optional
        Figure size for plots (width, height)
    
    Returns:
    --------
    dict : Dictionary containing analysis results
    """
    day_of_year, day_name = _parse_date_input(date_input)
    
    print(f"=== ANALYSIS FOR {day_name.upper()} (Day {day_of_year}) ===")
    print("=" * 60)
    
    # Get observed data for this day across all years
    day_data = data[data['day_of_year'] == day_of_year]['PRCP'].values
    
    if len(day_data) == 0:
        print(f"No observed data found for day {day_of_year}")
        return None
    
    # Calculate observed statistics
    obs_mean = np.mean(day_data)
    obs_std = np.std(day_data)
    obs_median = np.median(day_data)
    obs_rain_freq = np.mean(day_data > 0)
    obs_rainy_days = day_data[day_data > 0]
    obs_rainy_mean = np.mean(obs_rainy_days) if len(obs_rainy_days) > 0 else 0
    obs_max = np.max(day_data)
    obs_min = np.min(day_data)
    
    print(f"\nOBSERVED STATISTICS:")
    print(f"  Total observations: {len(day_data)}")
    print(f"  Rain frequency: {obs_rain_freq:.3f} ({obs_rain_freq*100:.1f}%)")
    print(f"  Mean rainfall: {obs_mean:.4f} mm")
    print(f"  Median rainfall: {obs_median:.4f} mm")
    print(f"  Std deviation: {obs_std:.4f} mm")
    print(f"  Min rainfall: {obs_min:.4f} mm")
    print(f"  Max rainfall: {obs_max:.4f} mm")
    if len(obs_rainy_days) > 0:
        print(f"  Mean on rainy days: {obs_rainy_mean:.4f} mm")
    
    # Use helper function to get model predictions
    n_samples = 1000
    rain_probs, expected_amounts, alpha_amounts = _evaluate_model_for_day(trace, day_of_year)
    
    # Limit to requested number of samples
    n_samples = min(n_samples, len(rain_probs))
    rain_probs = rain_probs[:n_samples]
    expected_amounts = expected_amounts[:n_samples]
    alpha_amounts = alpha_amounts[:n_samples]
    
    # Generate rainfall samples
    day_predictions = []
    for i, (p_rain, mu_amount, alpha_amount) in enumerate(zip(rain_probs, expected_amounts, alpha_amounts)):
        # Sample rain indicator
        rain_indicator = np.random.binomial(1, p_rain)
        
        # Sample rainfall amount if it rains
        if rain_indicator == 1:
            rainfall = np.random.gamma(alpha_amount, mu_amount / alpha_amount)
        else:
            rainfall = 0
        
        day_predictions.append(rainfall)
    
    day_predictions = np.array(day_predictions)
    rain_probabilities = rain_probs
    
    # Calculate predicted statistics
    pred_mean = np.mean(day_predictions)
    pred_std = np.std(day_predictions)
    pred_median = np.median(day_predictions)
    pred_rain_freq = np.mean(day_predictions > 0)
    pred_rainy_days = day_predictions[day_predictions > 0]
    pred_rainy_mean = np.mean(pred_rainy_days) if len(pred_rainy_days) > 0 else 0
    pred_max = np.max(day_predictions)
    pred_min = np.min(day_predictions)
    pred_rain_prob_mean = np.mean(rain_probabilities)
    pred_rain_prob_std = np.std(rain_probabilities)
    
    print(f"\nPREDICTED STATISTICS:")
    print(f"  Rain probability: {pred_rain_prob_mean:.3f} ± {pred_rain_prob_std:.3f}")
    print(f"  Rain frequency: {pred_rain_freq:.3f} ({pred_rain_freq*100:.1f}%)")
    print(f"  Mean rainfall: {pred_mean:.4f} mm")
    print(f"  Median rainfall: {pred_median:.4f} mm")
    print(f"  Std deviation: {pred_std:.4f} mm")
    print(f"  Min rainfall: {pred_min:.4f} mm")
    print(f"  Max rainfall: {pred_max:.4f} mm")
    if len(pred_rainy_days) > 0:
        print(f"  Mean on rainy days: {pred_rainy_mean:.4f} mm")
    
    # Model performance
    rain_freq_error = abs(obs_rain_freq - pred_rain_freq)
    rain_freq_error_pct = (rain_freq_error / obs_rain_freq) * 100 if obs_rain_freq > 0 else 0
    mean_error = abs(obs_mean - pred_mean)
    mean_error_pct = (mean_error / obs_mean) * 100 if obs_mean > 0 else 0
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"  Rain frequency error: {rain_freq_error:.3f} ({rain_freq_error_pct:.1f}%)")
    print(f"  Mean rainfall error: {mean_error:.4f} mm ({mean_error_pct:.1f}%)")
    
    # Confidence intervals
    pred_ci_95 = np.percentile(day_predictions, [2.5, 97.5])
    pred_ci_90 = np.percentile(day_predictions, [5, 95])
    pred_ci_50 = np.percentile(day_predictions, [25, 75])
    rain_prob_ci_95 = np.percentile(rain_probabilities, [2.5, 97.5])
    
    print(f"\nPREDICTION INTERVALS:")
    print(f"  Rainfall 95% CI: [{pred_ci_95[0]:.4f}, {pred_ci_95[1]:.4f}] mm")
    print(f"  Rainfall 90% CI: [{pred_ci_90[0]:.4f}, {pred_ci_90[1]:.4f}] mm")
    print(f"  Rainfall 50% CI: [{pred_ci_50[0]:.4f}, {pred_ci_50[1]:.4f}] mm")
    print(f"  Rain probability 95% CI: [{rain_prob_ci_95[0]:.3f}, {rain_prob_ci_95[1]:.3f}]")
    
    # Create plots if requested
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Distribution comparison
        ax1 = axes[0, 0]
        obs_df = pd.DataFrame({'Rainfall': day_data, 'Type': 'Observed'})
        pred_df = pd.DataFrame({'Rainfall': day_predictions, 'Type': 'Predicted'})
        combined_df = pd.concat([obs_df, pred_df], ignore_index=True)
        
        sns.histplot(data=combined_df, x='Rainfall', hue='Type', 
                    alpha=0.7, ax=ax1, bins=20, stat='density', 
                    multiple='dodge', common_norm=False)
        sns.rugplot(data=obs_df, x='Rainfall', ax=ax1, color='#1f77b4', alpha=0.5, height=0.05)
        
        ax1.axvline(obs_mean, color='#1f77b4', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Obs Mean: {obs_mean:.3f}')
        ax1.axvline(pred_mean, color='#ff7f0e', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Pred Mean: {pred_mean:.3f}')
        ax1.set_title(f'Rainfall Distribution: {day_name}')
        ax1.set_xlabel('Rainfall (mm)')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # 2. Rain probability distribution
        ax2 = axes[0, 1]
        ax2.hist(rain_probabilities, bins=30, alpha=0.7, color='green', density=True)
        ax2.axvline(pred_rain_prob_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {pred_rain_prob_mean:.3f}')
        ax2.axvline(obs_rain_freq, color='blue', linestyle='-', linewidth=2, 
                   label=f'Observed: {obs_rain_freq:.3f}')
        ax2.set_title(f'Rain Probability Distribution: {day_name}')
        ax2.set_xlabel('Rain Probability')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 3. Q-Q plot
        ax3 = axes[1, 0]
        obs_quantiles = np.percentile(day_data, np.linspace(0, 100, 100))
        pred_quantiles = np.percentile(day_predictions, np.linspace(0, 100, 100))
        ax3.scatter(obs_quantiles, pred_quantiles, alpha=0.6)
        ax3.plot([0, max(obs_quantiles)], [0, max(obs_quantiles)], 'r--', label='Perfect Match')
        ax3.set_xlabel('Observed Quantiles')
        ax3.set_ylabel('Predicted Quantiles')
        ax3.set_title(f'Q-Q Plot: {day_name}')
        ax3.legend()
        
        # 4. Time series comparison (if multiple years)
        ax4 = axes[1, 1]
        if len(day_data) > 1:
            years = data[data['day_of_year'] == day_of_year]['DATE'].dt.year.values
            ax4.scatter(years, day_data, alpha=0.7, label='Observed', s=50)
            ax4.axhline(pred_mean, color='red', linestyle='--', linewidth=2, 
                       label=f'Pred Mean: {pred_mean:.3f}')
            ax4.fill_between(years, pred_ci_95[0], pred_ci_95[1], alpha=0.3, color='red', 
                           label='95% CI')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Rainfall (mm)')
            ax4.set_title(f'Yearly Variation: {day_name}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor time series', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'Yearly Variation: {day_name}')
        
        plt.tight_layout()
        plt.show()
    
    # Return results dictionary
    results = {
        'day_of_year': day_of_year,
        'day_name': day_name,
        'observed': {
            'data': day_data,
            'mean': obs_mean,
            'std': obs_std,
            'median': obs_median,
            'rain_frequency': obs_rain_freq,
            'rainy_mean': obs_rainy_mean,
            'min': obs_min,
            'max': obs_max
        },
        'predicted': {
            'samples': day_predictions,
            'mean': pred_mean,
            'std': pred_std,
            'median': pred_median,
            'rain_frequency': pred_rain_freq,
            'rainy_mean': pred_rainy_mean,
            'min': pred_min,
            'max': pred_max,
            'rain_probability_mean': pred_rain_prob_mean,
            'rain_probability_std': pred_rain_prob_std,
            'ci_95': pred_ci_95,
            'ci_90': pred_ci_90,
            'ci_50': pred_ci_50,
            'rain_prob_ci_95': rain_prob_ci_95
        },
        'performance': {
            'rain_freq_error': rain_freq_error,
            'rain_freq_error_pct': rain_freq_error_pct,
            'mean_error': mean_error,
            'mean_error_pct': mean_error_pct
        }
    }
    
    return results


def calculate_rainfall_interval_probability(trace, date_input, interval_min=None, interval_max=None, 
                                          n_samples=1000):
    """
    Calculate the probability of rainfall being within a specified interval for a given day.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    model : pymc.Model
        PyMC model
    date_input : int, str, or tuple
        - int: day of year (1-365)
        - str: "MM/DD" format (e.g., "01/15")
        - tuple: (month, day) format (e.g., (1, 15))
    interval_min : float, optional
        Minimum rainfall amount (mm). If None, no lower bound.
    interval_max : float, optional
        Maximum rainfall amount (mm). If None, no upper bound.
    n_samples : int, optional
        Number of samples to use for calculation (default 1000)
    
    Returns:
    --------
    dict : Dictionary containing probability results
    
    Examples:
    --------
    # Any rainfall (no bounds)
    prob = calculate_rainfall_interval_probability(trace, "01/15")
    
    # Light rain (2.5 to 12.7 mm)
    prob = calculate_rainfall_interval_probability(trace, "01/15", 2.5, 12.7)
    
    # Heavy rain (25.4+ mm)
    prob = calculate_rainfall_interval_probability(trace, "01/15", 25.4)
    
    # Very light rain (up to 5.1 mm)
    prob = calculate_rainfall_interval_probability(trace, "01/15", None, 5.1)
    """
    day_of_year, day_name = _parse_date_input(date_input)
    
    # Use helper function to get model predictions
    rain_probs, expected_amounts, alpha_amounts = _evaluate_model_for_day(trace, day_of_year)
    
    # Limit to requested number of samples
    n_samples = min(n_samples, len(rain_probs))
    rain_probs = rain_probs[:n_samples]
    expected_amounts = expected_amounts[:n_samples]
    alpha_amounts = alpha_amounts[:n_samples]
    
    # Generate rainfall samples
    day_predictions = []
    for i, (p_rain, mu_amount, alpha_amount) in enumerate(zip(rain_probs, expected_amounts, alpha_amounts)):
        # Sample rain indicator
        rain_indicator = np.random.binomial(1, p_rain)
        
        # Sample rainfall amount if it rains
        if rain_indicator == 1:
            rainfall = np.random.gamma(alpha_amount, mu_amount / alpha_amount)
        else:
            rainfall = 0
        
        day_predictions.append(rainfall)
    
    day_predictions = np.array(day_predictions)
    
    # Handle different interval cases
    if interval_min is None and interval_max is None:
        # No bounds - probability of any rainfall > 0
        in_interval = day_predictions > 0
        interval_desc = "any rainfall (> 0 mm)"
    elif interval_min is None:
        # Only upper bound
        in_interval = day_predictions <= interval_max
        interval_desc = f"≤ {interval_max:.3f} mm"
    elif interval_max is None:
        # Only lower bound
        in_interval = day_predictions >= interval_min
        interval_desc = f"≥ {interval_min:.3f} mm"
    else:
        # Both bounds
        in_interval = (day_predictions >= interval_min) & (day_predictions <= interval_max)
        interval_desc = f"[{interval_min:.3f}, {interval_max:.3f}] mm"
    
    prob_in_interval = np.mean(in_interval)
    
    # Calculate confidence interval for the probability using binomial approximation
    n_samples = len(day_predictions)
    prob_std = np.sqrt(prob_in_interval * (1 - prob_in_interval) / n_samples)
    prob_ci_95 = [
        max(0, prob_in_interval - 1.96 * prob_std),
        min(1, prob_in_interval + 1.96 * prob_std)
    ]
    
    return {
        'day_of_year': day_of_year,
        'day_name': day_name,
        'interval_min': interval_min,
        'interval_max': interval_max,
        'interval_description': interval_desc,
        'probability': prob_in_interval,
        'probability_ci_95': prob_ci_95,
        'samples': day_predictions
    }


def print_rainfall_interval_probability(trace, date_input, interval_min=None, interval_max=None, 
                                       n_samples=1000):
    """
    Print the probability of rainfall being within a specified interval for a given day.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    model : pymc.Model
        PyMC model
    date_input : int, str, or tuple
        - int: day of year (1-365)
        - str: "MM/DD" format (e.g., "01/15")
        - tuple: (month, day) format (e.g., (1, 15))
    interval_min : float, optional
        Minimum rainfall amount (mm). If None, no lower bound.
    interval_max : float, optional
        Maximum rainfall amount (mm). If None, no upper bound.
    n_samples : int, optional
        Number of samples to use for calculation (default 1000)
    
    Returns:
    --------
    dict : Dictionary containing probability results
    """
    results = calculate_rainfall_interval_probability(trace, date_input, interval_min, interval_max, n_samples)
    
    print(f"=== RAINFALL INTERVAL PROBABILITY ANALYSIS ===")
    print(f"Date: {results['day_name']} (Day {results['day_of_year']})")
    print(f"Interval: {results['interval_description']}")
    print(f"Probability of rainfall in interval: {results['probability']:.3f} ({results['probability']*100:.1f}%)")
    print(f"95% CI for probability: [{results['probability_ci_95'][0]:.3f}, {results['probability_ci_95'][1]:.3f}]")
    
    return results


def calculate_any_rain_probability(trace, date_input, n_samples=1000):
    """
    Calculate the probability of any rain (rainfall > 0) on a given day.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    model : pymc.Model
        PyMC model
    date_input : int, str, or tuple
        - int: day of year (1-365)
        - str: "MM/DD" format (e.g., "01/15")
        - tuple: (month, day) format (e.g., (1, 15))
    n_samples : int, optional
        Number of samples to use for calculation (default 1000)
    
    Returns:
    --------
    dict : Dictionary containing probability results
    """
    day_of_year, day_name = _parse_date_input(date_input)
    
    # Use helper function to get model predictions
    rain_probs, _, _ = _evaluate_model_for_day(trace, day_of_year)
    
    # Limit to requested number of samples
    n_samples = min(n_samples, len(rain_probs))
    rain_probabilities = rain_probs[:n_samples]
    
    # Calculate statistics
    mean_prob = np.mean(rain_probabilities)
    std_prob = np.std(rain_probabilities)
    ci_95 = np.percentile(rain_probabilities, [2.5, 97.5])
    ci_90 = np.percentile(rain_probabilities, [5, 95])
    ci_50 = np.percentile(rain_probabilities, [25, 75])
    
    return {
        'day_of_year': day_of_year,
        'day_name': day_name,
        'mean_probability': mean_prob,
        'std_probability': std_prob,
        'ci_95': ci_95,
        'ci_90': ci_90,
        'ci_50': ci_50,
        'samples': rain_probabilities
    }


def print_any_rain_probability(trace, date_input, n_samples=1000):
    """
    Print the probability of any rain (rainfall > 0) on a given day.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    model : pymc.Model
        PyMC model
    date_input : int, str, or tuple
        - int: day of year (1-365)
        - str: "MM/DD" format (e.g., "01/15")
        - tuple: (month, day) format (e.g., (1, 15))
    n_samples : int, optional
        Number of samples to use for calculation (default 1000)
    
    Returns:
    --------
    dict : Dictionary containing probability results
    """
    results = calculate_any_rain_probability(trace, date_input, n_samples)
    
    print(f"=== ANY RAIN PROBABILITY ANALYSIS ===")
    print(f"Date: {results['day_name']} (Day {results['day_of_year']})")
    print(f"Probability of any rain: {results['mean_probability']:.3f} ± {results['std_probability']:.3f}")
    print(f"95% CI: [{results['ci_95'][0]:.3f}, {results['ci_95'][1]:.3f}]")
    print(f"90% CI: [{results['ci_90'][0]:.3f}, {results['ci_90'][1]:.3f}]")
    print(f"50% CI: [{results['ci_50'][0]:.3f}, {results['ci_50'][1]:.3f}]")
    
    return results


def print_simple_daily_rainfall_analysis(trace, date_input="01/15"):
    """
    Print example rainfall interval probabilities for a given date with confidence intervals.
    
    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from the Bayesian rainfall model.
    date_input : int, str, or tuple
        Date to analyze (e.g., "01/15", 15, or (1, 15)).
    """
    print("INTERVAL PROBABILITY EXAMPLES")
    print("=" * 60)
    print(f"Date analyzed: {date_input}\n")

    prob_any = calculate_rainfall_interval_probability(trace, date_input)
    ci_any = prob_any['probability_ci_95']
    print(f"Any rainfall: {prob_any['probability']:.3f} ± {(ci_any[1] - ci_any[0])/2:.3f}")
    print()

    prob_negligible = calculate_rainfall_interval_probability(trace, date_input, interval_max=0.1)
    ci_neg = prob_negligible['probability_ci_95']
    print(f"Negligible rain (<0.1 mm): {prob_negligible['probability']:.3f} ± {(ci_neg[1] - ci_neg[0])/2:.3f}")

    prob_light = calculate_rainfall_interval_probability(trace, date_input, interval_min=0.1, interval_max=2.5)
    ci_light = prob_light['probability_ci_95']
    print(f"Light rain (0.1–2.5 mm): {prob_light['probability']:.3f} ± {(ci_light[1] - ci_light[0])/2:.3f}")

    prob_moderate = calculate_rainfall_interval_probability(trace, date_input, interval_min=2.5, interval_max=10.0)
    ci_mod = prob_moderate['probability_ci_95']
    print(f"Moderate rain (2.5–10 mm): {prob_moderate['probability']:.3f} ± {(ci_mod[1] - ci_mod[0])/2:.3f}")

    prob_heavy = calculate_rainfall_interval_probability(trace, date_input, interval_min=10.0)
    ci_heavy = prob_heavy['probability_ci_95']
    print(f"Heavy rain (>10 mm): {prob_heavy['probability']:.3f} ± {(ci_heavy[1] - ci_heavy[0])/2:.3f}")

    print("\nNote: These should sum to 1.0 (with some rounding error)")
    total = prob_heavy['probability'] + prob_light['probability'] + prob_moderate['probability'] + prob_negligible['probability']
    print(f"Sum: {total:.3f}")
    
    print("\nConfidence intervals reflect uncertainty in the probability estimates due to parameter uncertainty.")


# =============================================================================
# YEAR-SPECIFIC ANALYSIS FUNCTIONS FOR HIERARCHICAL MODELS
# =============================================================================

def _evaluate_hierarchical_model_for_day(trace, day_of_year, year=None):
    """
    Helper function to evaluate hierarchical model for a specific day and year.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from hierarchical model sampling
    day_of_year : int
        Day of year (1-365)
    year : int, optional
        Year for year-specific prediction. If None, uses average year effects.
    
    Returns:
    --------
    tuple : (rain_probs, expected_amounts, alpha_amounts)
        Arrays of rain probabilities, expected rainfall amounts, and alpha parameters
    """
    # Get the number of harmonics from the trace
    n_harmonics = trace.posterior.a_rain.shape[-1]
    
    # Create harmonic features for the specific day (vectorized)
    h_values = np.arange(1, n_harmonics + 1)  # Shape: (n_harmonics,)
    day_sin = np.sin(2 * h_values * np.pi * day_of_year / 365.25)  # Shape: (n_harmonics,)
    day_cos = np.cos(2 * h_values * np.pi * day_of_year / 365.25)  # Shape: (n_harmonics,)
    
    # Get parameter samples
    a_rain = trace.posterior.a_rain.values  # Shape: (chains, draws, n_harmonics)
    b_rain = trace.posterior.b_rain.values
    c_rain = trace.posterior.c_rain.values  # Shape: (chains, draws)
    
    a_amount = trace.posterior.a_amount.values
    b_amount = trace.posterior.b_amount.values
    c_amount = trace.posterior.c_amount.values
    
    alpha_amount = trace.posterior.alpha_amount.values  # Shape: (chains, draws)
    
    # Get year effects
    year_rain_effects = trace.posterior.year_rain_effects.values  # Shape: (chains, draws, n_years)
    year_amount_effects = trace.posterior.year_amount_effects.values
    
    # Get unique years from the model
    unique_years = trace.posterior.year_rain_effects.coords['year'].values
    
    # Handle year selection
    if year is not None:
        if year not in unique_years:
            raise ValueError(f"Year {year} not found in model. Available years: {unique_years}")
        year_idx = np.where(unique_years == year)[0][0]
        year_rain_effect = year_rain_effects[:, :, year_idx]  # Shape: (chains, draws)
        year_amount_effect = year_amount_effects[:, :, year_idx]
    else:
        # Use average year effects (all zeros since they're centered)
        year_rain_effect = np.zeros_like(c_rain)
        year_amount_effect = np.zeros_like(c_amount)
    
    # Calculate rain probability for all samples
    # Reshape for broadcasting: (chains, draws, n_harmonics) @ (n_harmonics,) -> (chains, draws)
    harmonic_rain_contribution = np.sum(
        a_rain * day_sin[None, None, :] + b_rain * day_cos[None, None, :], 
        axis=-1
    )
    
    logit_p = c_rain + harmonic_rain_contribution + year_rain_effect
    rain_probs = 1 / (1 + np.exp(-logit_p))  # Sigmoid
    
    # Calculate expected rainfall amount for all samples
    harmonic_amount_contribution = np.sum(
        a_amount * day_sin[None, None, :] + b_amount * day_cos[None, None, :], 
        axis=-1
    )
    
    log_mu_amount = c_amount + harmonic_amount_contribution + year_amount_effect
    expected_amounts = np.exp(log_mu_amount)
    
    # Flatten to 1D array
    rain_probs = rain_probs.flatten()
    expected_amounts = expected_amounts.flatten()
    alpha_amounts = alpha_amount.flatten()
    
    return rain_probs, expected_amounts, alpha_amounts




def analyze_single_day_hierarchical(trace, data, date_input, year=None, show_plots=True):
    """
    Comprehensive analysis for a specific day using hierarchical model with year effects.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from hierarchical model
    data : pd.DataFrame
        Original data (for comparison)
    date_input : int, str, or tuple
        Day specification (see _parse_date_input for formats)
    year : int, optional
        Year for year-specific prediction. If None, uses average year effects.
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict : Analysis results
    """
    day_of_year, day_name = _parse_date_input(date_input)
    
    # Get model predictions
    rain_probs, expected_amounts, alpha_amounts = _evaluate_hierarchical_model_for_day(trace, day_of_year, year)
    
    # Calculate statistics
    rain_prob_mean = rain_probs.mean()
    rain_prob_std = rain_probs.std()
    rain_prob_ci = np.percentile(rain_probs, [2.5, 97.5])
    
    expected_amount_mean = expected_amounts.mean()
    expected_amount_std = expected_amounts.std()
    expected_amount_ci = np.percentile(expected_amounts, [2.5, 97.5])
    
    # Get observed data for this day
    day_data = data[data['day_of_year'] == day_of_year]
    if year is not None:
        day_data = day_data[day_data['year'] == year]
    
    observed_rain_freq = day_data['PRCP'].gt(0).mean() if len(day_data) > 0 else np.nan
    observed_mean_amount = day_data[day_data['PRCP'] > 0]['PRCP'].mean() if len(day_data) > 0 else np.nan
    
    # Print results
    year_str = f" for {year}" if year is not None else " (average year effects)"
    print(f"HIERARCHICAL MODEL ANALYSIS: {day_name}{year_str}")
    print("=" * 60)
    print(f"Rain Probability: {rain_prob_mean:.3f} ± {rain_prob_std:.3f}")
    print(f"95% CI: [{rain_prob_ci[0]:.3f}, {rain_prob_ci[1]:.3f}]")
    print(f"Expected Amount: {expected_amount_mean:.2f} ± {expected_amount_std:.2f} mm")
    print(f"95% CI: [{expected_amount_ci[0]:.2f}, {expected_amount_ci[1]:.2f}] mm")
    
    if not np.isnan(observed_rain_freq):
        print(f"Observed Rain Frequency: {observed_rain_freq:.3f}")
    if not np.isnan(observed_mean_amount):
        print(f"Observed Mean Amount: {observed_mean_amount:.2f} mm")
    
    # Create plots if requested
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Hierarchical Model Analysis: {day_name}{year_str}", fontsize=14)
        
        # Rain probability distribution
        axes[0, 0].hist(rain_probs, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(rain_prob_mean, color='red', linestyle='--', label=f'Mean: {rain_prob_mean:.3f}')
        axes[0, 0].axvline(rain_prob_ci[0], color='orange', linestyle=':', alpha=0.7, label='95% CI')
        axes[0, 0].axvline(rain_prob_ci[1], color='orange', linestyle=':', alpha=0.7)
        if not np.isnan(observed_rain_freq):
            axes[0, 0].axvline(observed_rain_freq, color='green', linestyle='-', linewidth=2, label=f'Observed: {observed_rain_freq:.3f}')
        axes[0, 0].set_xlabel('Rain Probability')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Rain Probability Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Expected amount distribution
        axes[0, 1].hist(expected_amounts, bins=50, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(expected_amount_mean, color='red', linestyle='--', label=f'Mean: {expected_amount_mean:.2f}')
        axes[0, 1].axvline(expected_amount_ci[0], color='orange', linestyle=':', alpha=0.7, label='95% CI')
        axes[0, 1].axvline(expected_amount_ci[1], color='orange', linestyle=':', alpha=0.7)
        if not np.isnan(observed_mean_amount):
            axes[0, 1].axvline(observed_mean_amount, color='green', linestyle='-', linewidth=2, label=f'Observed: {observed_mean_amount:.2f}')
        axes[0, 1].set_xlabel('Expected Amount (mm)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Expected Rainfall Amount Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Alpha parameter distribution
        axes[1, 0].hist(alpha_amounts, bins=50, alpha=0.7, density=True, color='lightcoral', edgecolor='black')
        axes[1, 0].axvline(alpha_amounts.mean(), color='red', linestyle='--', label=f'Mean: {alpha_amounts.mean():.2f}')
        axes[1, 0].set_xlabel('Alpha Parameter')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Gamma Shape Parameter (Alpha)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Year effects (if available)
        if hasattr(trace.posterior, 'year_rain_effects'):
            year_effects = trace.posterior.year_rain_effects.values.flatten()
            axes[1, 1].hist(year_effects, bins=30, alpha=0.7, density=True, color='plum', edgecolor='black')
            axes[1, 1].axvline(0, color='red', linestyle='--', label='Zero (average)')
            if year is not None:
                year_idx = np.where(trace.posterior.year_rain_effects.coords['year'].values == year)[0][0]
                specific_year_effect = trace.posterior.year_rain_effects.values[:, :, year_idx].flatten()
                axes[1, 1].axvline(specific_year_effect.mean(), color='green', linestyle='-', linewidth=2, label=f'Year {year}: {specific_year_effect.mean():.3f}')
            axes[1, 1].set_xlabel('Year Effect')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Year Effects Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No year effects available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Year Effects')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'day_of_year': day_of_year,
        'day_name': day_name,
        'year': year,
        'rain_probability': {
            'mean': rain_prob_mean,
            'std': rain_prob_std,
            'ci_lower': rain_prob_ci[0],
            'ci_upper': rain_prob_ci[1]
        },
        'expected_amount': {
            'mean': expected_amount_mean,
            'std': expected_amount_std,
            'ci_lower': expected_amount_ci[0],
            'ci_upper': expected_amount_ci[1]
        },
        'observed': {
            'rain_frequency': observed_rain_freq,
            'mean_amount': observed_mean_amount
        }
    }


def compare_years_for_day(trace, data, date_input, years=None):
    """
    Compare predictions for a specific day across different years.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from hierarchical model
    data : pd.DataFrame
        Original data
    date_input : int, str, or tuple
        Day specification
    years : list of int, optional
        Years to compare. If None, uses all available years.
        
    Returns:
    --------
    dict : Comparison results
    """
    day_of_year, day_name = _parse_date_input(date_input)
    
    if years is None:
        years = trace.posterior.year_rain_effects.coords['year'].values
    
    results = {}
    
    print(f"YEAR COMPARISON FOR {day_name}")
    print("=" * 50)
    print(f"{'Year':<6} {'P(Rain)':<12} {'Expected (mm)':<15} {'Observed P(Rain)':<18} {'Observed Mean (mm)':<20}")
    print("-" * 80)
    
    for year in years:
        # Get predictions
        rain_probs, expected_amounts, _ = _evaluate_hierarchical_model_for_day(trace, day_of_year, year)
        
        # Get observed data
        year_data = data[(data['day_of_year'] == day_of_year) & (data['year'] == year)]
        observed_rain_freq = year_data['PRCP'].gt(0).mean() if len(year_data) > 0 else np.nan
        observed_mean_amount = year_data[year_data['PRCP'] > 0]['PRCP'].mean() if len(year_data) > 0 else np.nan
        
        # Store results
        results[year] = {
            'rain_probability': {
                'mean': rain_probs.mean(),
                'std': rain_probs.std()
            },
            'expected_amount': {
                'mean': expected_amounts.mean(),
                'std': expected_amounts.std()
            },
            'observed': {
                'rain_frequency': observed_rain_freq,
                'mean_amount': observed_mean_amount
            }
        }
        
        # Print results
        obs_rain_str = f"{observed_rain_freq:.3f}" if not np.isnan(observed_rain_freq) else "N/A"
        obs_amount_str = f"{observed_mean_amount:.2f}" if not np.isnan(observed_mean_amount) else "N/A"
        
        print(f"{year:<6} {rain_probs.mean():.3f}±{rain_probs.std():.3f}    {expected_amounts.mean():.2f}±{expected_amounts.std():.2f}      {obs_rain_str:<18} {obs_amount_str:<20}")
    
    return results


def plot_year_effects(trace, data):
    """
    Plot year effects for both rain probability and rainfall amount.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from hierarchical model
    data : pd.DataFrame
        Original data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Year Effects Analysis', fontsize=16)
    
    # Get year effects
    year_rain_effects = trace.posterior.year_rain_effects.values  # (chains, draws, n_years)
    year_amount_effects = trace.posterior.year_amount_effects.values
    years = trace.posterior.year_rain_effects.coords['year'].values
    
    # Rain probability year effects
    rain_means = year_rain_effects.mean(axis=(0, 1))
    rain_stds = year_rain_effects.std(axis=(0, 1))
    rain_cis = np.percentile(year_rain_effects, [2.5, 97.5], axis=(0, 1))
    
    # Convert years to datetime for proper x-axis
    year_dates = pd.to_datetime(years, format='%Y')
    
    axes[0, 0].errorbar(year_dates, rain_means, yerr=rain_stds, fmt='o-', capsize=5, capthick=2)
    axes[0, 0].fill_between(year_dates, rain_cis[0], rain_cis[1], alpha=0.3)
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_ylabel('Year Effect (logit scale)')
    axes[0, 0].set_title('Rain Probability Year Effects')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Format x-axis with year labels
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Rainfall amount year effects
    amount_means = year_amount_effects.mean(axis=(0, 1))
    amount_stds = year_amount_effects.std(axis=(0, 1))
    amount_cis = np.percentile(year_amount_effects, [2.5, 97.5], axis=(0, 1))
    
    axes[0, 1].errorbar(year_dates, amount_means, yerr=amount_stds, fmt='o-', capsize=5, capthick=2, color='green')
    axes[0, 1].fill_between(year_dates, amount_cis[0], amount_cis[1], alpha=0.3, color='green')
    axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_ylabel('Year Effect (log scale)')
    axes[0, 1].set_title('Rainfall Amount Year Effects')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Format x-axis with year labels
    axes[0, 1].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Observed vs predicted rain frequency by year
    observed_rain_freq = []
    predicted_rain_freq = []
    
    for year in years:
        year_data = data[data['year'] == year]
        if len(year_data) > 0:
            obs_freq = year_data['PRCP'].gt(0).mean()
            # Use average year effects for prediction
            pred_freq = 1 / (1 + np.exp(-year_rain_effects[:, :, years == year].mean()))
            observed_rain_freq.append(obs_freq)
            predicted_rain_freq.append(pred_freq)
    
    axes[1, 0].scatter(observed_rain_freq, predicted_rain_freq, s=100, alpha=0.7)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.7)
    axes[1, 0].set_xlabel('Observed Rain Frequency')
    axes[1, 0].set_ylabel('Predicted Rain Frequency')
    axes[1, 0].set_title('Observed vs Predicted Rain Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Year effect magnitudes
    rain_effect_magnitudes = np.abs(year_rain_effects).mean(axis=(0, 1))
    amount_effect_magnitudes = np.abs(year_amount_effects).mean(axis=(0, 1))
    
    x = np.arange(len(years))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, rain_effect_magnitudes, width, label='Rain Probability', alpha=0.7)
    axes[1, 1].bar(x + width/2, amount_effect_magnitudes, width, label='Rainfall Amount', alpha=0.7)
    axes[1, 1].set_ylabel('Effect Magnitude')
    axes[1, 1].set_title('Year Effect Magnitudes')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(years)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# HIERARCHICAL MODEL HELPER FUNCTIONS
# =============================================================================

def _get_year_effects_from_trace(trace):
    """Extract and flatten year effects from trace."""
    year_rain_effects = trace.posterior.year_rain_effects.values  # Shape: (chains, draws, n_years)
    year_amount_effects = trace.posterior.year_amount_effects.values
    
    # Handle both real xarray coords and mock coords
    try:
        unique_years = trace.posterior.year_rain_effects.coords['year'].values
    except AttributeError:
        # Handle mock coords
        unique_years = trace.posterior.year_rain_effects.coords['year']
    
    # Flatten to get all samples
    year_rain_effects_flat = year_rain_effects.reshape(-1, len(unique_years))  # Shape: (n_samples, n_years)
    year_amount_effects_flat = year_amount_effects.reshape(-1, len(unique_years))
    
    return year_rain_effects_flat, year_amount_effects_flat, unique_years


def _apply_year_effects_to_predictions(rain_probs_base, expected_amounts_base, 
                                     year_rain_effects_flat, year_amount_effects_flat, unique_years):
    """Apply sampled year effects to base predictions."""
    rain_probs_with_year = []
    expected_amounts_with_year = []
    
    for i in range(len(rain_probs_base)):
        # Sample a random year for this posterior sample
        year_idx = np.random.choice(len(unique_years))
        year_rain_effect = year_rain_effects_flat[i, year_idx]
        year_amount_effect = year_amount_effects_flat[i, year_idx]
        
        # Apply year effects to base predictions
        # For rain probability: add to logit
        logit_p = np.log(rain_probs_base[i] / (1 - rain_probs_base[i])) + year_rain_effect
        rain_prob_with_year = 1 / (1 + np.exp(-logit_p))
        
        # For rainfall amount: add to log scale
        log_mu_amount = np.log(expected_amounts_base[i]) + year_amount_effect
        expected_amount_with_year = np.exp(log_mu_amount)
        
        rain_probs_with_year.append(rain_prob_with_year)
        expected_amounts_with_year.append(expected_amount_with_year)
    
    return np.array(rain_probs_with_year), np.array(expected_amounts_with_year)


def _calculate_expected_values_with_year_effects(trace, days_of_year):
    """Calculate expected values with sampled year effects (vectorized version)."""
    # Get year effects from trace
    year_rain_effects_flat, year_amount_effects_flat, unique_years = _get_year_effects_from_trace(trace)
    n_samples = year_rain_effects_flat.shape[0]
    n_days = len(days_of_year)
    n_years = len(unique_years)
    n_harmonics = trace.posterior.a_rain.values.shape[-1]
    
    # Get all parameters at once (flattened)
    a_rain = trace.posterior.a_rain.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    b_rain = trace.posterior.b_rain.values.reshape(-1, n_harmonics)
    c_rain = trace.posterior.c_rain.values.reshape(-1, 1)
    a_amount = trace.posterior.a_amount.values.reshape(-1, n_harmonics)
    b_amount = trace.posterior.b_amount.values.reshape(-1, n_harmonics)
    c_amount = trace.posterior.c_amount.values.reshape(-1, 1)
    alpha_amount = trace.posterior.alpha_amount.values.reshape(-1, 1)
    
    # Sample year effects for all posterior samples at once
    year_indices = np.random.choice(n_years, size=n_samples)
    year_rain_effects = year_rain_effects_flat[np.arange(n_samples), year_indices]  # Shape: (n_samples,)
    year_amount_effects = year_amount_effects_flat[np.arange(n_samples), year_indices]  # Shape: (n_samples,)
    
    # Vectorize harmonic calculations for ALL days at once
    # Shape: (n_harmonics, n_days)
    h_values = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
    day_values = days_of_year[None, :]  # Shape: (1, n_days)
    
    # Vectorized harmonic calculations
    day_sin_all = np.sin(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
    day_cos_all = np.cos(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
    
    # Vectorize rain probability calculations for ALL days at once
    # Shape: (n_samples, n_days)
    harmonic_rain_contrib = np.sum(a_rain[:, :, None] * day_sin_all[None, :, :] + 
                                  b_rain[:, :, None] * day_cos_all[None, :, :], axis=1)
    logit_p = c_rain + harmonic_rain_contrib + year_rain_effects[:, None]
    rain_probs = 1 / (1 + np.exp(-logit_p))
    
    # Vectorize rainfall amount calculations for ALL days at once
    # Shape: (n_samples, n_days)
    harmonic_amount_contrib = np.sum(a_amount[:, :, None] * day_sin_all[None, :, :] + 
                                   b_amount[:, :, None] * day_cos_all[None, :, :], axis=1)
    log_mu_amount = c_amount + harmonic_amount_contrib + year_amount_effects[:, None]
    expected_amounts = np.exp(log_mu_amount)
    
    # Alpha amounts are the same for all days (no harmonic dependence)
    # Shape: (n_samples, n_days)
    alpha_amounts = np.broadcast_to(alpha_amount, (n_samples, n_days))
    
    return rain_probs.T, expected_amounts.T, alpha_amounts.T  # Shape: (n_days, n_samples)


def _get_all_year_predictions(trace, days_of_year, unique_years):
    """Get year-specific predictions for all days and years at once (vectorized version)."""
    # Get the actual number of samples from the trace (flattened)
    n_samples = trace.posterior.a_rain.values.size // trace.posterior.a_rain.values.shape[-1]
    n_days = len(days_of_year)
    n_years = len(unique_years)
    n_harmonics = trace.posterior.a_rain.values.shape[-1]
    
    # Get all parameters at once (flattened)
    a_rain = trace.posterior.a_rain.values.reshape(-1, n_harmonics)  # Shape: (n_samples, n_harmonics)
    b_rain = trace.posterior.b_rain.values.reshape(-1, n_harmonics)
    c_rain = trace.posterior.c_rain.values.reshape(-1, 1)
    a_amount = trace.posterior.a_amount.values.reshape(-1, n_harmonics)
    b_amount = trace.posterior.b_amount.values.reshape(-1, n_harmonics)
    c_amount = trace.posterior.c_amount.values.reshape(-1, 1)
    
    # Get year effects
    year_rain_effects = trace.posterior.year_rain_effects.values.reshape(-1, n_years)  # Shape: (n_samples, n_years)
    year_amount_effects = trace.posterior.year_amount_effects.values.reshape(-1, n_years)
    
    # Initialize arrays
    all_year_rain_probs = np.zeros((n_samples, n_days, n_years))
    all_year_expected_amounts = np.zeros((n_samples, n_days, n_years))
    
    # Vectorize harmonic calculations for ALL days at once
    # Shape: (n_harmonics, n_days)
    h_values = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
    day_values = days_of_year[None, :]  # Shape: (1, n_days)
    
    # Vectorized harmonic calculations
    day_sin_all = np.sin(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
    day_cos_all = np.cos(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
    
    # Calculate for each year
    for k, year in enumerate(unique_years):
        # Get year effects for this year
        year_rain_effect = year_rain_effects[:, k:k+1]  # Shape: (n_samples, 1)
        year_amount_effect = year_amount_effects[:, k:k+1]  # Shape: (n_samples, 1)
        
        # Vectorize rain probability calculations for ALL days at once
        # Shape: (n_samples, n_days)
        harmonic_rain_contrib = np.sum(a_rain[:, :, None] * day_sin_all[None, :, :] + 
                                      b_rain[:, :, None] * day_cos_all[None, :, :], axis=1)
        logit_p = c_rain + harmonic_rain_contrib + year_rain_effect
        rain_probs_year = 1 / (1 + np.exp(-logit_p))
        
        # Vectorize rainfall amount calculations for ALL days at once
        # Shape: (n_samples, n_days)
        harmonic_amount_contrib = np.sum(a_amount[:, :, None] * day_sin_all[None, :, :] + 
                                       b_amount[:, :, None] * day_cos_all[None, :, :], axis=1)
        log_mu_amount = c_amount + harmonic_amount_contrib + year_amount_effect
        expected_amounts_year = np.exp(log_mu_amount)
        
        all_year_rain_probs[:, :, k] = rain_probs_year
        all_year_expected_amounts[:, :, k] = expected_amounts_year
    
    return all_year_rain_probs, all_year_expected_amounts


def _sample_posterior_predictive_hierarchical(trace, data, days_of_year, expected_amounts, alpha_amounts):
    """
    Sample posterior predictive for hierarchical model (efficient vectorized version).
    
    This function samples actual rainfall observations that might occur, incorporating:
    1. Parameter uncertainty (from MCMC samples)
    2. Year-to-year variation (by sampling one random year per posterior sample)
    3. Natural stochastic variation (binary rain + gamma amounts)
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from hierarchical model
    data : pandas.DataFrame  
        Original data (used to get unique years)
    days_of_year : array
        Days of year (1-365)
    expected_amounts : array
        Pre-computed expected amounts (shape: n_days, n_samples)
    alpha_amounts : array
        Pre-computed alpha parameters (shape: n_days, n_samples)
        
    Returns:
    --------
    tuple : (rain_indicators, rainfall_amounts)
        Arrays of shape (n_samples, n_days) representing possible observations
    """
    print(f"  Starting efficient hierarchical posterior predictive sampling...")
    
    unique_years = data['year'].unique()
    n_years = len(unique_years)
    n_samples = expected_amounts.shape[1]
    n_days = len(days_of_year)
    
    print(f"  Setup: {n_samples} posterior samples, {n_years} unique years")
    
    # Get year effects from trace (flattened)
    year_rain_effects = trace.posterior.year_rain_effects.values.reshape(-1, n_years)  # (n_samples, n_years)
    year_amount_effects = trace.posterior.year_amount_effects.values.reshape(-1, n_years)
    
    print(f"  Sampling one random year per posterior sample...")
    # Sample ONE random year for each posterior sample (not 100!)
    # Shape: (n_samples,)
    sampled_year_indices = np.random.choice(n_years, size=n_samples)
    
    print(f"  Extracting year effects for sampled years...")
    # Get year effects for sampled years
    # Shape: (n_samples,)
    sample_indices = np.arange(n_samples)
    sampled_year_rain_effects = year_rain_effects[sample_indices, sampled_year_indices]
    sampled_year_amount_effects = year_amount_effects[sample_indices, sampled_year_indices]
    
    print(f"  Computing rain probabilities with year effects...")
    # Apply year effects to get rain probabilities
    # expected_amounts shape: (n_days, n_samples) -> need to transpose for broadcasting
    base_logit_p = np.log(expected_amounts.T / (1 - expected_amounts.T + 1e-8))  # Avoid log(0)
    # Actually, we need to recompute logit_p from the base parameters since expected_amounts is already exponentiated
    
    # Get base parameters (flattened)
    n_harmonics = trace.posterior.a_rain.shape[-1]
    a_rain = trace.posterior.a_rain.values.reshape(-1, n_harmonics)  # (n_samples, n_harmonics)
    b_rain = trace.posterior.b_rain.values.reshape(-1, n_harmonics)
    c_rain = trace.posterior.c_rain.values.flatten()  # (n_samples,)
    a_amount = trace.posterior.a_amount.values.reshape(-1, n_harmonics)
    b_amount = trace.posterior.b_amount.values.reshape(-1, n_harmonics)
    c_amount = trace.posterior.c_amount.values.flatten()
    
    # Compute harmonic features for all days (vectorized)
    h_values = np.arange(1, n_harmonics + 1)[:, None]  # (n_harmonics, 1)
    day_values = days_of_year[None, :]  # (1, n_days)
    day_sin_all = np.sin(2 * h_values * np.pi * day_values / 365.25)  # (n_harmonics, n_days)
    day_cos_all = np.cos(2 * h_values * np.pi * day_values / 365.25)
    
    # Compute rain probabilities with sampled year effects
    # Shape: (n_samples, n_days)
    harmonic_rain_contrib = np.sum(a_rain[:, :, None] * day_sin_all[None, :, :] + 
                                  b_rain[:, :, None] * day_cos_all[None, :, :], axis=1)
    logit_p = c_rain[:, None] + harmonic_rain_contrib + sampled_year_rain_effects[:, None]
    rain_probs = 1 / (1 + np.exp(-logit_p))
    
    # Compute expected amounts with sampled year effects  
    # Shape: (n_samples, n_days)
    harmonic_amount_contrib = np.sum(a_amount[:, :, None] * day_sin_all[None, :, :] + 
                                   b_amount[:, :, None] * day_cos_all[None, :, :], axis=1)
    log_mu_amount = c_amount[:, None] + harmonic_amount_contrib + sampled_year_amount_effects[:, None]
    mu_amounts = np.exp(log_mu_amount)
    
    print(f"  Sampling binary rain outcomes...")
    # Sample binary rain outcomes
    # Shape: (n_samples, n_days)
    rain_indicators = np.random.binomial(1, rain_probs)
    
    print(f"  Sampling rainfall amounts for rainy days...")
    # Sample rainfall amounts for rainy days only
    # Shape: (n_samples, n_days)
    rainfall_amounts = np.zeros_like(rain_probs)
    rainy_mask = rain_indicators == 1
    
    if np.any(rainy_mask):
        # Get alpha values (same for all days in this model)
        alpha_vals = trace.posterior.alpha_amount.values.flatten()[:n_samples]  # (n_samples,)
        
        # Expand alpha to match the shape of mu_amounts for vectorized sampling
        alpha_expanded = np.broadcast_to(alpha_vals[:, None], (n_samples, n_days))  # (n_samples, n_days)
        
    # Vectorized gamma sampling for rainy days
        rainfall_amounts[rainy_mask] = np.random.gamma(
            alpha_expanded[rainy_mask],  # Shape parameter
            mu_amounts[rainy_mask] / alpha_expanded[rainy_mask]  # Scale parameter
        )
    
    print(f"  Final shapes: rain_indicators {rain_indicators.shape}, rainfall_amounts {rainfall_amounts.shape}")
    print(f"  Efficient hierarchical posterior predictive sampling completed!")
    
    return rain_indicators, rainfall_amounts


def _get_observed_data(data):
    """Extract observed data for plotting."""
    observed_rain_prob = data.groupby('day_of_year')['PRCP'].apply(lambda x: (x > 0).mean()).values
    observed_days = data.groupby('day_of_year')['PRCP'].apply(lambda x: (x > 0).mean()).index.values
    rainy_data = data[data['PRCP'] > 0]
    observed_rainfall = rainy_data.groupby('day_of_year')['PRCP'].mean().values
    observed_rainy_days = rainy_data.groupby('day_of_year')['PRCP'].mean().index.values
    
    return observed_rain_prob, observed_days, observed_rainfall, observed_rainy_days
