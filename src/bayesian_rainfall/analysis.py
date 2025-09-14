"""
Analysis functions for Bayesian rainfall model results.
"""

import numpy as np
import pandas as pd
import arviz as az


def print_model_summary(trace, data, param_names=None):
    """
    Print a comprehensive summary of the model results including statistics,
    convergence diagnostics, and seasonal patterns.
    
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
