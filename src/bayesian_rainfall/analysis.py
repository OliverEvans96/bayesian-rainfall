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
    print(f"Mean rainfall on rainy days: {data[data['PRCP'] > 0]['PRCP'].mean():.3f} mm")
    print(f"Max rainfall: {data['PRCP'].max():.3f} mm")

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
    
    # Get model predictions for this day
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Sample predictions
    n_samples = 1000
    day_predictions = []
    rain_probabilities = []
    
    for i in range(min(n_samples, len(trace.posterior.a_rain.values.flatten()))):
        chain_idx = i // trace.posterior.a_rain.shape[1]
        sample_idx = i % trace.posterior.a_rain.shape[1]
        
        a_rain = trace.posterior.a_rain.values[chain_idx, sample_idx]
        b_rain = trace.posterior.b_rain.values[chain_idx, sample_idx]
        c_rain = trace.posterior.c_rain.values[chain_idx, sample_idx]
        a_amount = trace.posterior.a_amount.values[chain_idx, sample_idx]
        b_amount = trace.posterior.b_amount.values[chain_idx, sample_idx]
        c_amount = trace.posterior.c_amount.values[chain_idx, sample_idx]
        alpha_amount = trace.posterior.alpha_amount.values[chain_idx, sample_idx]
        
        # Calculate rain probability
        logit_p = c_rain + a_rain * day_sin + b_rain * day_cos
        p_rain = 1 / (1 + np.exp(-logit_p))
        rain_probabilities.append(p_rain)
        
        # Sample rain indicator
        rain_indicator = np.random.binomial(1, p_rain)
        
        # Sample rainfall amount if it rains
        if rain_indicator == 1:
            mu_amount = np.exp(c_amount + a_amount * day_sin + b_amount * day_cos)
            rainfall = np.random.gamma(alpha_amount, mu_amount / alpha_amount)
        else:
            rainfall = 0
        
        day_predictions.append(rainfall)
    
    day_predictions = np.array(day_predictions)
    rain_probabilities = np.array(rain_probabilities)
    
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
    
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Sample predictions
    day_predictions = []
    
    for i in range(min(n_samples, len(trace.posterior.a_rain.values.flatten()))):
        chain_idx = i // trace.posterior.a_rain.shape[1]
        sample_idx = i % trace.posterior.a_rain.shape[1]
        
        a_rain = trace.posterior.a_rain.values[chain_idx, sample_idx]
        b_rain = trace.posterior.b_rain.values[chain_idx, sample_idx]
        c_rain = trace.posterior.c_rain.values[chain_idx, sample_idx]
        a_amount = trace.posterior.a_amount.values[chain_idx, sample_idx]
        b_amount = trace.posterior.b_amount.values[chain_idx, sample_idx]
        c_amount = trace.posterior.c_amount.values[chain_idx, sample_idx]
        alpha_amount = trace.posterior.alpha_amount.values[chain_idx, sample_idx]
        
        # Calculate rain probability
        logit_p = c_rain + a_rain * day_sin + b_rain * day_cos
        p_rain = 1 / (1 + np.exp(-logit_p))
        
        # Sample rain indicator
        rain_indicator = np.random.binomial(1, p_rain)
        
        # Sample rainfall amount if it rains
        if rain_indicator == 1:
            mu_amount = np.exp(c_amount + a_amount * day_sin + b_amount * day_cos)
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
    
    # Calculate confidence interval for the probability
    prob_samples = []
    for _ in range(1000):  # Bootstrap samples
        bootstrap_sample = np.random.choice(day_predictions, size=len(day_predictions), replace=True)
        if interval_min is None and interval_max is None:
            bootstrap_in_interval = bootstrap_sample > 0
        elif interval_min is None:
            bootstrap_in_interval = bootstrap_sample <= interval_max
        elif interval_max is None:
            bootstrap_in_interval = bootstrap_sample >= interval_min
        else:
            bootstrap_in_interval = (bootstrap_sample >= interval_min) & (bootstrap_sample <= interval_max)
        prob_samples.append(np.mean(bootstrap_in_interval))
    
    prob_ci_95 = np.percentile(prob_samples, [2.5, 97.5])
    
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
    
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Sample rain probabilities
    rain_probabilities = []
    
    for i in range(min(n_samples, len(trace.posterior.a_rain.values.flatten()))):
        chain_idx = i // trace.posterior.a_rain.shape[1]
        sample_idx = i % trace.posterior.a_rain.shape[1]
        
        a_rain = trace.posterior.a_rain.values[chain_idx, sample_idx]
        b_rain = trace.posterior.b_rain.values[chain_idx, sample_idx]
        c_rain = trace.posterior.c_rain.values[chain_idx, sample_idx]
        
        # Calculate rain probability
        logit_p = c_rain + a_rain * day_sin + b_rain * day_cos
        p_rain = 1 / (1 + np.exp(-logit_p))
        rain_probabilities.append(p_rain)
    
    rain_probabilities = np.array(rain_probabilities)
    
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
    Print example rainfall interval probabilities for a given date.
    
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

    # Case 1: No bounds (any rainfall)
    prob_any = calculate_rainfall_interval_probability(trace, date_input)
    print(f"Any rainfall: {prob_any['probability']:.3f}")

    prob_negligible = calculate_rainfall_interval_probability(trace, date_input, interval_max=0.1)
    print(f"Negligible rain (<0.1 mm): {prob_negligible['probability']:.3f}")

    # Case 2: Heavy rain (>10 mm)
    prob_heavy = calculate_rainfall_interval_probability(trace, date_input, interval_min=10.0)
    print(f"Heavy rain (>10 mm): {prob_heavy['probability']:.3f}")

    # Case 3: Light rain (0.1–2.5 mm)
    prob_light = calculate_rainfall_interval_probability(trace, date_input, interval_min=0.1, interval_max=2.5)
    print(f"Light rain (0.1–2.5 mm): {prob_light['probability']:.3f}")

    # Case 4: Moderate rain (2.5–10 mm)
    prob_moderate = calculate_rainfall_interval_probability(trace, date_input, interval_min=2.5, interval_max=10.0)
    print(f"Moderate rain (2.5–10 mm): {prob_moderate['probability']:.3f}")

    print("\nNote: These should sum to 1.0 (with some rounding error)")
    total = prob_heavy['probability'] + prob_light['probability'] + prob_moderate['probability'] + prob_negligible['probability']
    print(f"Sum: {total:.3f}")
