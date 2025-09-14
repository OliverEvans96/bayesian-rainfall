"""
Analysis functions for Bayesian rainfall model results.
"""

import numpy as np
import pandas as pd
import arviz as az


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
    
    # Calculate seasonal patterns for rain probability and amounts
    days_of_year = np.arange(1, 366)
    day_sin_pred = np.sin(2 * np.pi * days_of_year / 365.25)
    day_cos_pred = np.cos(2 * np.pi * days_of_year / 365.25)
    
    # Get posterior samples
    a_rain_samples = trace.posterior.a_rain.values.flatten()
    b_rain_samples = trace.posterior.b_rain.values.flatten()
    c_rain_samples = trace.posterior.c_rain.values.flatten()
    a_amount_samples = trace.posterior.a_amount.values.flatten()
    b_amount_samples = trace.posterior.b_amount.values.flatten()
    c_amount_samples = trace.posterior.c_amount.values.flatten()
    
    # Calculate rain probabilities
    rain_probs = []
    for i in range(len(a_rain_samples)):
        logit_p = c_rain_samples[i] + a_rain_samples[i] * day_sin_pred + b_rain_samples[i] * day_cos_pred
        p_rain = 1 / (1 + np.exp(-logit_p))
        rain_probs.append(p_rain)
    
    rain_probs = np.array(rain_probs)
    rain_prob_mean = np.mean(rain_probs, axis=0)
    
    # Calculate rainfall amounts
    rainfall_amounts = []
    for i in range(len(a_amount_samples)):
        mu_amount = np.exp(c_amount_samples[i] + a_amount_samples[i] * day_sin_pred + b_amount_samples[i] * day_cos_pred)
        rainfall_amounts.append(mu_amount)
    
    rainfall_amounts = np.array(rainfall_amounts)
    rainfall_mean = np.mean(rainfall_amounts, axis=0)
    
    print("=== MODEL SUMMARY ===")
    print(f"Total observations: {len(data)}")
    print(f"Rainy days: {len(data[data['PRCP'] > 0])}")
    print(f"Overall rain frequency: {len(data[data['PRCP'] > 0]) / len(data):.3f}")
    print(f"Mean rainfall on rainy days: {data[data['PRCP'] > 0]['PRCP'].mean():.3f} inches")
    print(f"Max rainfall: {data['PRCP'].max():.3f} inches")

    print("\n=== CONVERGENCE DIAGNOSTICS ===")
    print("R-hat values (should be < 1.01):")
    for param in param_names:
        rhat = az.rhat(trace, var_names=[param])[param].values
        print(f"  {param}: {rhat:.4f}")

    print("\n=== MODEL FIT ANALYSIS ===")
    # Calculate model predictions for observed days
    observed_days = data['day_of_year'].values
    day_sin_obs = np.sin(2 * np.pi * observed_days / 365.25)
    day_cos_obs = np.cos(2 * np.pi * observed_days / 365.25)
    
    # Calculate predicted rain probabilities for observed days
    predicted_rain_probs = []
    for i in range(len(a_rain_samples)):
        logit_p = c_rain_samples[i] + a_rain_samples[i] * day_sin_obs + b_rain_samples[i] * day_cos_obs
        p_rain = 1 / (1 + np.exp(-logit_p))
        predicted_rain_probs.append(p_rain)
    
    predicted_rain_probs = np.array(predicted_rain_probs)
    predicted_rain_prob_mean = np.mean(predicted_rain_probs, axis=0)
    
    # Calculate predicted rainfall amounts for observed days
    predicted_rainfall_amounts = []
    for i in range(len(a_amount_samples)):
        mu_amount = np.exp(c_amount_samples[i] + a_amount_samples[i] * day_sin_obs + b_amount_samples[i] * day_cos_obs)
        predicted_rainfall_amounts.append(mu_amount)
    
    predicted_rainfall_amounts = np.array(predicted_rainfall_amounts)
    predicted_rainfall_mean = np.mean(predicted_rainfall_amounts, axis=0)
    
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
            month_sin = np.sin(2 * np.pi * month_days / 365.25)
            month_cos = np.cos(2 * np.pi * month_days / 365.25)
            
            month_rain_probs = []
            for i in range(len(a_rain_samples)):
                logit_p = c_rain_samples[i] + a_rain_samples[i] * month_sin + b_rain_samples[i] * month_cos
                p_rain = 1 / (1 + np.exp(-logit_p))
                month_rain_probs.append(p_rain)
            
            month_rain_probs = np.array(month_rain_probs)
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
    print(f"Peak rainfall amount: Day {peak_amount_day} ({rainfall_mean[np.argmax(rainfall_mean)]:.3f} inches)")
    print(f"Minimum rainfall amount: Day {min_amount_day} ({rainfall_mean[np.argmin(rainfall_mean)]:.3f} inches)")


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
