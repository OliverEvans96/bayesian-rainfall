"""
Visualization functions for Bayesian rainfall analysis.
"""

import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from .analysis import (
    _evaluate_model_for_day,
    _get_year_effects_from_trace,
    _apply_year_effects_to_predictions,
    _calculate_expected_values_with_year_effects,
    _get_all_year_predictions,
    _sample_posterior_predictive_hierarchical,
    _get_observed_data
)


def _day_of_year_to_dates(days_of_year, year=2024):
    """
    Convert day-of-year values to proper datetime objects for plotting.
    
    Parameters:
    -----------
    days_of_year : array-like
        Day of year values (1-365)
    year : int, optional
        Year to use for date conversion (default 2024)
    
    Returns:
    --------
    array : Array of datetime objects
    """
    dates = []
    for day in days_of_year:
        # Handle leap year
        if day == 365 and not pd.Timestamp(f"{year}-12-31").is_leap_year:
            # If it's not a leap year, day 365 should be Dec 31
            date = datetime(year, 12, 31)
        else:
            date = datetime(year, 1, 1) + timedelta(days=int(day) - 1)
        dates.append(date)
    return np.array(dates)


def _sample_observed_rain_frequencies(rain_probs, n_years=6):
    """
    Sample observed rain frequencies accounting for sampling uncertainty using Beta distribution.
    
    This function models the sampling uncertainty from observing only n_years of data per day,
    using a Beta distribution to approximate the sampling distribution of the sample proportion.
    
    Parameters:
    -----------
    rain_probs : array (n_days, n_samples)
        Posterior samples of true rain probabilities
    n_years : int
        Number of years of observations per day
        
    Returns:
    --------
    observed_frequencies : array (n_days, n_samples)
        Sampled observed frequencies including sampling uncertainty
    """
    n_days, n_samples = rain_probs.shape
    observed_frequencies = np.zeros_like(rain_probs)
    
    for d in range(n_days):
        for i in range(n_samples):
            p_i = rain_probs[d, i]
            
            # Calculate Beta parameters to match sampling distribution
            # For observed frequency X/n where X ~ Binomial(n, p_i):
            # Mean = p_i, Variance = p_i * (1 - p_i) / n
            mu = p_i
            sigma_sq = p_i * (1 - p_i) / n_years
            
            if 0 < p_i < 1 and sigma_sq > 0:
                # Method of moments: match mean and variance
                scale_factor = mu * (1 - mu) / sigma_sq - 1
                if scale_factor > 0:  # Valid Beta parameters
                    alpha = mu * scale_factor
                    beta = (1 - mu) * scale_factor
                    observed_frequencies[d, i] = np.random.beta(alpha, beta)
                else:
                    observed_frequencies[d, i] = p_i  # Fallback to mean
            else:
                observed_frequencies[d, i] = p_i  # Edge case fallback
    
    return observed_frequencies


def plot_trace(trace, param_names=None, figsize=(15, 8)):
    """
    Plot MCMC trace plots to check convergence.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    param_names : list, optional
        List of parameter names to plot. If None, uses main parameters.
    figsize : tuple, optional
        Figure size (width, height)
    """
    if param_names is None:
        param_names = ['a_rain', 'b_rain', 'c_rain', 'a_amount', 'b_amount', 'c_amount', 'alpha_amount']
    
    az.plot_trace(trace, var_names=param_names, figsize=figsize)
    plt.tight_layout()
    plt.show()


def _create_combined_plots(days_of_year, rain_prob_mean, rain_prob_lower, rain_prob_upper,
                         pp_rain_mean, pp_rain_lower, pp_rain_upper,
                         rainfall_mean, rainfall_lower, rainfall_upper,
                         pp_amounts_mean, pp_amounts_lower, pp_amounts_upper,
                         observed_rain_prob, observed_days, observed_rainfall, observed_rainy_days,
                         ci_level, figsize, n_years):
    """Create the combined plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Convert day-of-year to dates for proper x-axis
    dates = _day_of_year_to_dates(days_of_year)
    observed_dates = _day_of_year_to_dates(observed_days)
    observed_rainy_dates = _day_of_year_to_dates(observed_rainy_days)
    
    # Rain probability plot
    ax1.fill_between(dates, rain_prob_lower, rain_prob_upper, alpha=0.2, color='blue', 
                    label=f'{int(ci_level*100)}% CI (Parameter Uncertainty)')
    ax1.plot(dates, rain_prob_mean, 'b-', linewidth=2, label='Expected Probability')
    ax1.fill_between(dates, pp_rain_lower, pp_rain_upper, alpha=0.3, color='green', 
                    label=f'{int(ci_level*100)}% CI (Observed Frequencies, n={n_years})')
    ax1.plot(dates, pp_rain_mean, 'g--', linewidth=2, label='Mean Observed Frequency')
    ax1.scatter(observed_dates, observed_rain_prob, alpha=0.6, color='red', s=20, label='Observed')
    ax1.set_ylabel('Rain Probability')
    ax1.set_title('Rain Probability Predictions Across the Year')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis with month labels
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    # Rainfall amount plot
    ax2.fill_between(dates, rainfall_lower, rainfall_upper, alpha=0.2, color='blue', 
                    label=f'{int(ci_level*100)}% CI (Expected Amount)')
    ax2.plot(dates, rainfall_mean, 'b-', linewidth=2, label='Expected Amount')
    ax2.fill_between(dates, pp_amounts_lower, pp_amounts_upper, alpha=0.3, color='orange',
                    label='95% CI (Posterior Predictive)')
    ax2.plot(dates, pp_amounts_mean, 'orange', linestyle='--', linewidth=2, label='Posterior Predictive Mean')
    ax2.scatter(observed_rainy_dates, observed_rainfall, alpha=0.6, color='red', s=20, label='Observed')
    ax2.set_ylabel('Expected Rainfall (mm)')
    ax2.set_title('Rainfall Amount Predictions Across the Year')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis with month labels
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    plt.tight_layout()
    plt.show()


def plot_combined_predictions(trace, data, ci_level=0.95, figsize=(14, 10)):
    """
    Plot combined rain probability and amount predictions across the year.
    
    Shows two types of confidence intervals:
    - Blue CI: Parameter uncertainty only (expected values)
    - Green CI: Observed frequencies accounting for sampling uncertainty (based on actual data years)
    
    For rain probability:
    - Blue: Range of expected rain probabilities given parameter uncertainty
    - Green: Range of observed rain frequencies we might see with the actual number of years of data
    
    For rainfall amounts:
    - Blue: Range of expected amounts given parameter uncertainty  
    - Green: Range of possible rainfall observations (including stochastic variation)
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    data : pandas.DataFrame
        Weather data with columns 'day_of_year' and 'PRCP'
    ci_level : float, optional
        Credible interval level (default 0.95)
    figsize : tuple, optional
        Figure size (width, height)
    """
    print("Starting plot_combined_predictions...")
    
    # Calculate number of years from data
    n_years = data['year'].nunique()
    print(f"Detected {n_years} years of data: {sorted(data['year'].unique())}")
    
    # Calculate percentiles
    lower_percentile = 50 - (ci_level * 50)
    upper_percentile = 50 + (ci_level * 50)
    
    days_of_year = np.arange(1, 366)
    
    print("Calculating expected values with year effects...")
    # Calculate expected values with year effects
    rain_probs, expected_amounts, alpha_amounts = _calculate_expected_values_with_year_effects(trace, days_of_year)
    
    print("Calculating statistics for expected values...")
    # Calculate statistics for expected values
    rain_prob_mean = np.mean(rain_probs, axis=1)
    rain_prob_lower = np.percentile(rain_probs, lower_percentile, axis=1)
    rain_prob_upper = np.percentile(rain_probs, upper_percentile, axis=1)
    
    print("Sampling posterior predictive (efficient version)...")
    # Sample posterior predictive
    rain_indicators, rainfall_amounts = _sample_posterior_predictive_hierarchical(
        trace, data, days_of_year, expected_amounts, alpha_amounts
    )
    
    print(f"Posterior predictive samples: rain_indicators shape {rain_indicators.shape}, rainfall_amounts shape {rainfall_amounts.shape}")
    
    print("Calculating posterior predictive statistics...")
    # Calculate posterior predictive statistics
    
    # For rain probability PP CI: sample observed frequencies with sampling uncertainty
    # This shows what rain frequencies we might observe given the actual number of years of data
    print(f"Sampling observed rain frequencies with sampling uncertainty (n={n_years} years)...")
    observed_rain_frequencies = _sample_observed_rain_frequencies(rain_probs, n_years=n_years)
    
    pp_rain_mean = np.mean(observed_rain_frequencies, axis=1)  # Average across samples for each day
    pp_rain_lower = np.percentile(observed_rain_frequencies, lower_percentile, axis=1)
    pp_rain_upper = np.percentile(observed_rain_frequencies, upper_percentile, axis=1)
    
    pp_amounts_mean = np.mean(rainfall_amounts, axis=0)  # Average across samples for each day
    pp_amounts_lower = np.percentile(rainfall_amounts, lower_percentile, axis=0)
    pp_amounts_upper = np.percentile(rainfall_amounts, upper_percentile, axis=0)
    
    print("Calculating parameter uncertainty statistics...")
    # Calculate expected values (parameter uncertainty)
    rainfall_mean = np.mean(expected_amounts, axis=1)
    rainfall_lower = np.percentile(expected_amounts, lower_percentile, axis=1)
    rainfall_upper = np.percentile(expected_amounts, upper_percentile, axis=1)
    
    print("Getting observed data...")
    # Get observed data
    observed_rain_prob, observed_days, observed_rainfall, observed_rainy_days = _get_observed_data(data)
    
    print("Creating plots...")
    # Create plots
    _create_combined_plots(days_of_year, rain_prob_mean, rain_prob_lower, rain_prob_upper,
                         pp_rain_mean, pp_rain_lower, pp_rain_upper,
                         rainfall_mean, rainfall_lower, rainfall_upper,
                         pp_amounts_mean, pp_amounts_lower, pp_amounts_upper,
                         observed_rain_prob, observed_days, observed_rainfall, observed_rainy_days,
                         ci_level, figsize, n_years)
    
    print("Plot completed!")


def plot_posterior_predictive_checks(trace, data, n_samples=100, figsize=(15, 10)):
    """
    Plot posterior predictive checks comparing observed vs predicted data.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    data : pandas.DataFrame
        Weather data with columns 'PRCP' and 'day_of_year'
    n_samples : int, optional
        Number of posterior samples to use for predictions
    figsize : tuple, optional
        Figure size (width, height)
    """
    # Generate full rainfall predictions using direct evaluation
    full_rainfall_predictions = []
    
    # Get parameter samples
    a_rain_samples = trace.posterior.a_rain.values.flatten()
    b_rain_samples = trace.posterior.b_rain.values.flatten()
    c_rain_samples = trace.posterior.c_rain.values.flatten()
    a_amount_samples = trace.posterior.a_amount.values.flatten()
    b_amount_samples = trace.posterior.b_amount.values.flatten()
    c_amount_samples = trace.posterior.c_amount.values.flatten()
    alpha_amount_samples = trace.posterior.alpha_amount.values.flatten()
    
    n_samples = min(n_samples, len(a_rain_samples))
    
    # Vectorize all calculations across samples
    day_of_year = data['day_of_year'].values
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Vectorize rain probability calculations for all samples
    # Shape: (n_samples, n_days)
    logit_p = (c_rain_samples[:, None] + 
               a_rain_samples[:, None] * day_sin[None, :] + 
               b_rain_samples[:, None] * day_cos[None, :])
    p_rain = 1 / (1 + np.exp(-logit_p))
    
    # Vectorize rain indicator sampling for all samples
    # Shape: (n_samples, n_days)
    rain_indicators = np.random.binomial(1, p_rain)
    
    # Vectorize expected rainfall amount calculations for all samples
    # Shape: (n_samples, n_days)
    mu_amount = np.exp(c_amount_samples[:, None] + 
                      a_amount_samples[:, None] * day_sin[None, :] + 
                      b_amount_samples[:, None] * day_cos[None, :])
    
    # Vectorize rainfall amount sampling for all samples
    # Shape: (n_samples, n_days)
    rainfall = np.zeros((n_samples, len(day_of_year)))
    rainy_mask = rain_indicators == 1
    
    # Only sample rainfall for rainy days (vectorized)
    if np.any(rainy_mask):
        rainfall[rainy_mask] = np.random.gamma(
            alpha_amount_samples[:, None][rainy_mask],
            mu_amount[rainy_mask] / alpha_amount_samples[:, None][rainy_mask]
        )
    
    full_rainfall_predictions = rainfall
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Add month column if not present
    if 'month' not in data.columns:
        data['month'] = pd.to_datetime(data['DATE']).dt.month
    
    # 1. Rain frequency by month
    monthly_rain_freq_obs = data.groupby('month')['PRCP'].apply(lambda x: (x > 0).mean())
    monthly_rain_freq_pred = []
    for month in range(1, 13):
        month_data = data[data['month'] == month]
        if len(month_data) > 0:
            month_indices = month_data.index - data.index[0]
            month_predictions = full_rainfall_predictions[:, month_indices]
            month_rain_freq = np.mean(month_predictions > 0, axis=1)
            monthly_rain_freq_pred.append(month_rain_freq)
    
    monthly_rain_freq_pred = np.array(monthly_rain_freq_pred).T
    
    axes[0, 0].boxplot(monthly_rain_freq_pred, positions=range(1, 13), widths=0.6)
    axes[0, 0].plot(range(1, 13), monthly_rain_freq_obs, 'ro-', label='Observed')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Rain Frequency')
    axes[0, 0].set_title('Monthly Rain Frequency: Observed vs Predicted')
    axes[0, 0].legend()
    
    # 2. Rainfall amount distribution
    rainy_obs = data[data['PRCP'] > 0]['PRCP'].values
    rainy_pred = full_rainfall_predictions[full_rainfall_predictions > 0].flatten()
    
    axes[0, 1].hist(rainy_obs, bins=30, alpha=0.7, label='Observed', density=True)
    axes[0, 1].hist(rainy_pred, bins=30, alpha=0.7, label='Predicted', density=True)
    axes[0, 1].set_xlabel('Rainfall Amount (mm)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Rainfall Amount Distribution')
    axes[0, 1].legend()
    
    # 3. Time series comparison (first 100 days)
    first_100_days = data.head(100)
    first_100_pred = full_rainfall_predictions[:, :100]
    
    # Convert dates for proper x-axis
    first_100_dates = pd.to_datetime(first_100_days['DATE'])
    
    axes[1, 0].plot(first_100_dates, first_100_days['PRCP'].values, 'b-', label='Observed', alpha=0.7)
    axes[1, 0].plot(first_100_dates, np.mean(first_100_pred, axis=0), 'r-', label='Predicted Mean', alpha=0.7)
    axes[1, 0].fill_between(first_100_dates, 
                           np.percentile(first_100_pred, 2.5, axis=0),
                           np.percentile(first_100_pred, 97.5, axis=0),
                           alpha=0.3, color='red', label='95% CI')
    axes[1, 0].set_ylabel('Rainfall (mm)')
    axes[1, 0].set_title('Time Series: First 100 Days')
    axes[1, 0].legend()
    
    # Format x-axis with month labels
    axes[1, 0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[1, 0].xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    # 4. Q-Q plot for rainfall amounts
    obs_quantiles = np.percentile(rainy_obs, np.linspace(0, 100, 100))
    pred_quantiles = np.percentile(rainy_pred, np.linspace(0, 100, 100))
    
    axes[1, 1].scatter(obs_quantiles, pred_quantiles, alpha=0.6)
    axes[1, 1].plot([0, max(obs_quantiles)], [0, max(obs_quantiles)], 'r--', label='Perfect Match')
    axes[1, 1].set_xlabel('Observed Quantiles')
    axes[1, 1].set_ylabel('Predicted Quantiles')
    axes[1, 1].set_title('Q-Q Plot: Rainfall Amounts')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_specific_days_comparison(trace, data, selected_days=None, day_names=None, figsize=(16, 12)):
    """
    Plot observed vs predicted rainfall distributions for specific days.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    data : pandas.DataFrame
        Weather data with columns 'day_of_year' and 'PRCP'
    selected_days : list, optional
        List of day numbers to plot. Default is [15, 100, 200, 300]
    day_names : list, optional
        List of day names for display. Default is ['Jan 15', 'Apr 10', 'Jul 19', 'Oct 27']
    figsize : tuple, optional
        Figure size (width, height)
    """
    if selected_days is None:
        selected_days = [15, 100, 200, 300]
    if day_names is None:
        day_names = ['Jan 15', 'Apr 10', 'Jul 19', 'Oct 27']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Pre-compute all day predictions vectorized
    all_day_predictions = []
    all_day_data = []
    
    for i, (day, day_name) in enumerate(zip(selected_days, day_names)):
        # Get observed data for this day of year across all years
        day_data = data[data['day_of_year'] == day]['PRCP'].values
        all_day_data.append(day_data)
        
        # Get predicted data for this day of year using the helper function
        rain_probs, expected_amounts, alpha_amounts = _evaluate_model_for_day(trace, day)
        
        # Vectorize sampling for this specific day
        n_samples = min(100, len(rain_probs))
        
        # Sample rain indicators for all samples at once
        rain_indicators = np.random.binomial(1, rain_probs[:n_samples])
        
        # Sample rainfall amounts for all samples at once
        rainfall = np.zeros(n_samples)
        rainy_mask = rain_indicators == 1
        if np.any(rainy_mask):
            rainfall[rainy_mask] = np.random.gamma(
                alpha_amounts[:n_samples][rainy_mask],
                expected_amounts[:n_samples][rainy_mask] / alpha_amounts[:n_samples][rainy_mask]
            )
        
        all_day_predictions.append(rainfall)
    
    # Process each day for plotting
    for i, (day, day_name) in enumerate(zip(selected_days, day_names)):
        day_data = all_day_data[i]
        day_predictions = all_day_predictions[i]
        
        # Create data for seaborn
        obs_df = pd.DataFrame({'Rainfall': day_data, 'Type': 'Observed'})
        pred_df = pd.DataFrame({'Rainfall': day_predictions, 'Type': 'Predicted'})
        combined_df = pd.concat([obs_df, pred_df], ignore_index=True)
        
        # Plot grouped histograms with seaborn
        sns.histplot(data=combined_df, x='Rainfall', hue='Type', 
                    alpha=0.7, ax=axes[i], bins=20, stat='density', common_norm=False, 
                    multiple='dodge')
        
        # Add rug plot for observed data
        sns.rugplot(data=obs_df, x='Rainfall', ax=axes[i], color='#1f77b4', alpha=0.5, height=0.05)
        
        # Add statistics
        obs_mean = np.mean(day_data)
        pred_mean = np.mean(day_predictions)
        obs_std = np.std(day_data)
        pred_std = np.std(day_predictions)
        
        # Add mean lines
        axes[i].axvline(obs_mean, color='#1f77b4', linestyle='--', alpha=0.8, linewidth=2, 
                       label=f'Obs Mean: {obs_mean:.3f}')
        axes[i].axvline(pred_mean, color='#ff7f0e', linestyle='--', alpha=0.8, linewidth=2, 
                       label=f'Pred Mean: {pred_mean:.3f}')
        
        axes[i].set_title(f'{day_name} (Day {day})', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Rainfall (mm)', fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].legend(fontsize=10)
        
        # Add text box with statistics
        textstr = f'Obs: μ={obs_mean:.3f}, σ={obs_std:.3f}\nPred: μ={pred_mean:.3f}, σ={pred_std:.3f}'
        axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Observed vs Predicted Rainfall Distributions for Specific Days', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_seasonal_summaries(trace, data, figsize=(16, 12)):
    """
    Plot seasonal comparison of observed vs predicted rainfall distributions.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    data : pandas.DataFrame
        Weather data with columns 'PRCP', 'day_of_year', and 'month'
    figsize : tuple, optional
        Figure size (width, height)
    """
    # Define seasons
    seasons = {
        'Winter': [1, 2, 12],  # Dec, Jan, Feb
        'Spring': [3, 4, 5],   # Mar, Apr, May
        'Summer': [6, 7, 8],   # Jun, Jul, Aug
        'Fall': [9, 10, 11]    # Sep, Oct, Nov
    }
    
    # Add month column if not present
    if 'month' not in data.columns:
        data['month'] = pd.to_datetime(data['DATE']).dt.month
    
    # Generate full rainfall predictions for all data
    full_rainfall_predictions = []
    n_samples = 100
    
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
        
        # Calculate rain probabilities
        day_of_year = data['day_of_year'].values
        day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
        
        logit_p = c_rain + a_rain * day_sin + b_rain * day_cos
        p_rain = 1 / (1 + np.exp(-logit_p))
        
        # Sample rain indicators
        rain_indicator = np.random.binomial(1, p_rain)
        
        # Calculate expected rainfall amounts
        mu_amount = np.exp(c_amount + a_amount * day_sin + b_amount * day_cos)
        
        # Sample rainfall amounts for rainy days
        rainfall = np.zeros(len(day_of_year))
        rainy_mask = rain_indicator == 1
        if np.sum(rainy_mask) > 0:
            rainfall[rainy_mask] = np.random.gamma(alpha_amount, mu_amount[rainy_mask] / alpha_amount)
        
        full_rainfall_predictions.append(rainfall)
    
    full_rainfall_predictions = np.array(full_rainfall_predictions)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, (season_name, months) in enumerate(seasons.items()):
        # Get observed data for this season
        season_data = data[data['month'].isin(months)]['PRCP'].values
        
        # Get predicted data for this season
        season_mask = data['month'].isin(months)
        season_predictions = full_rainfall_predictions[:, season_mask]
        season_predictions = season_predictions.flatten()
        
        # Create data for seaborn
        obs_df = pd.DataFrame({'Rainfall': season_data, 'Type': 'Observed'})
        pred_df = pd.DataFrame({'Rainfall': season_predictions, 'Type': 'Predicted'})
        combined_df = pd.concat([obs_df, pred_df], ignore_index=True)
        
        # Plot grouped histograms with seaborn
        sns.histplot(data=combined_df, x='Rainfall', hue='Type', 
                    alpha=0.7, ax=axes[i], bins=25, stat='density', 
                    multiple='dodge', common_norm=False)
        
        # Add rug plot for observed data
        sns.rugplot(data=obs_df, x='Rainfall', ax=axes[i], color='#1f77b4', alpha=0.5, height=0.05)
        
        # Add statistics
        obs_mean = np.mean(season_data)
        pred_mean = np.mean(season_predictions)
        obs_std = np.std(season_data)
        pred_std = np.std(season_predictions)
        
        # Calculate rain frequency
        obs_rain_freq = np.mean(season_data > 0)
        pred_rain_freq = np.mean(season_predictions > 0)
        
        # Add mean lines
        axes[i].axvline(obs_mean, color='#1f77b4', linestyle='--', alpha=0.8, linewidth=2, 
                       label=f'Obs Mean: {obs_mean:.3f}')
        axes[i].axvline(pred_mean, color='#ff7f0e', linestyle='--', alpha=0.8, linewidth=2, 
                       label=f'Pred Mean: {pred_mean:.3f}')
        
        axes[i].set_title(f'{season_name} Season', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Rainfall (mm)', fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].legend(fontsize=10)
        
        # Add text box with statistics
        textstr = f'Obs: μ={obs_mean:.3f}, σ={obs_std:.3f}, P(rain)={obs_rain_freq:.3f}\nPred: μ={pred_mean:.3f}, σ={pred_std:.3f}, P(rain)={pred_rain_freq:.3f}'
        axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Seasonal Comparison: Observed vs Predicted Rainfall Distributions', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_weekly_rain_probability(trace, data, figsize=(16, 8)):
    """
    Plot the chance of any rain each week throughout the year.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    data : pandas.DataFrame
        Weather data with columns 'day_of_year' and 'PRCP'
    figsize : tuple, optional
        Figure size (width, height)
    """
    # Create weekly bins (52 weeks)
    weeks = np.arange(1, 53)
    week_centers = np.arange(1, 366, 7)  # Center of each week
    
    # Calculate rain probabilities for each week using the existing helper function
    weekly_rain_probs = []
    
    for week in weeks:
        # Use the center day of the week for calculation
        day_of_year = week * 7 - 3  # Center of the week
        rain_probs, _, _ = _evaluate_model_for_day(trace, day_of_year)
        weekly_rain_probs.append(rain_probs)
    
    weekly_rain_probs = np.array(weekly_rain_probs)
    
    # Calculate statistics for expected probability
    mean_probs = np.mean(weekly_rain_probs, axis=1)
    std_probs = np.std(weekly_rain_probs, axis=1)
    lower_ci = np.percentile(weekly_rain_probs, 2.5, axis=1)
    upper_ci = np.percentile(weekly_rain_probs, 97.5, axis=1)
    
    # Generate posterior predictive samples for each week
    n_samples = 1000
    posterior_predictive_weekly = []
    
    for week in weeks:
        day_of_year = week * 7 - 3  # Center of the week
        rain_probs, _, _ = _evaluate_model_for_day(trace, day_of_year)
        
        # Sample binary rain indicators for this week
        week_predictions = np.random.binomial(1, rain_probs[:n_samples])
        posterior_predictive_weekly.append(week_predictions)
    
    posterior_predictive_weekly = np.array(posterior_predictive_weekly)
    
    # Calculate posterior predictive statistics
    pp_mean = np.mean(posterior_predictive_weekly, axis=1)
    pp_lower_ci = np.percentile(posterior_predictive_weekly, 2.5, axis=1)
    pp_upper_ci = np.percentile(posterior_predictive_weekly, 97.5, axis=1)
    
    # Calculate observed rain probabilities by week
    data['week'] = ((data['day_of_year'] - 1) // 7) + 1
    observed_weekly_probs = data.groupby('week')['PRCP'].apply(lambda x: (x > 0).mean())
    
    # Ensure we have probabilities for all 52 weeks (vectorized)
    observed_weekly_probs_full = np.zeros(52)
    week_indices = np.arange(1, 53)
    valid_weeks = observed_weekly_probs.index.values
    mask = np.isin(week_indices, valid_weeks)
    observed_weekly_probs_full[mask] = observed_weekly_probs[week_indices[mask]].values
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]//2))
    
    # Convert weeks to dates for proper x-axis
    week_dates = _day_of_year_to_dates(week_centers)
    
    # Weekly rain probability with confidence intervals
    ax.fill_between(week_dates, lower_ci, upper_ci, alpha=0.2, color='blue', 
                    label='95% CI (Expected Probability)')
    ax.plot(week_dates, mean_probs, 'b-', linewidth=2, label='Expected Probability')
    ax.fill_between(week_dates, pp_lower_ci, pp_upper_ci, alpha=0.3, color='green', 
                    label='95% CI (Posterior Predictive)')
    ax.plot(week_dates, pp_mean, 'g--', linewidth=2, label='Posterior Predictive Mean')
    ax.scatter(week_dates, observed_weekly_probs_full, color='red', s=30, alpha=0.7, 
               label='Observed Probability', zorder=5)
    ax.set_ylabel('Probability of Any Rain')
    ax.set_title('Weekly Rain Probability Throughout the Year', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Format x-axis with month labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("WEEKLY RAIN PROBABILITY SUMMARY")
    print("=" * 50)
    print(f"Peak rain probability: Week {np.argmax(mean_probs) + 1} ({np.max(mean_probs):.3f})")
    print(f"Minimum rain probability: Week {np.argmin(mean_probs) + 1} ({np.min(mean_probs):.3f})")
    
    return {
        'weeks': weeks,
        'expected_probs': mean_probs,
        'expected_lower_ci': lower_ci,
        'expected_upper_ci': upper_ci,
        'posterior_predictive_probs': pp_mean,
        'pp_lower_ci': pp_lower_ci,
        'pp_upper_ci': pp_upper_ci,
        'observed_probs': observed_weekly_probs_full
    }


def plot_hierarchical_posterior_predictive_checks(trace, data, n_samples=100, figsize=(15, 10)):
    """
    Plot posterior predictive checks for hierarchical model comparing observed vs predicted data.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from hierarchical model sampling
    data : pandas.DataFrame
        Weather data with columns 'PRCP', 'day_of_year', and 'year'
    n_samples : int, optional
        Number of posterior samples to use for predictions
    figsize : tuple, optional
        Figure size (width, height)
    """
    from .analysis import _evaluate_model_for_day
    
    # Generate posterior predictive samples for each day
    days_of_year = data['day_of_year'].values
    years = data['year'].values
    unique_days = np.unique(days_of_year)
    
    # Store predictions for each day
    pp_rain_probs = []
    pp_rainfall_amounts = []
    
    for day in unique_days:
        # Get all years for this day
        day_mask = days_of_year == day
        day_years = years[day_mask]
        
        # Sample from posterior for each year
        day_rain_probs = []
        day_rainfall_amounts = []
        
        for year in day_years:
            # Get posterior samples for this day and year
            rain_probs, expected_amounts, alpha_amounts = _evaluate_model_for_day(trace, day, year)
            
            # Sample n_samples from the posterior
            n_available = len(rain_probs)
            if n_available >= n_samples:
                sample_indices = np.random.choice(n_available, n_samples, replace=False)
            else:
                sample_indices = np.random.choice(n_available, n_samples, replace=True)
            
            day_rain_probs.extend(rain_probs[sample_indices])
            day_rainfall_amounts.extend(expected_amounts[sample_indices])
        
        # Convert to numpy arrays and store
        pp_rain_probs.append(np.array(day_rain_probs))
        pp_rainfall_amounts.append(np.array(day_rainfall_amounts))
    
    # Calculate statistics for each day
    pp_rain_mean = []
    pp_rain_lower = []
    pp_rain_upper = []
    pp_amount_mean = []
    pp_amount_lower = []
    pp_amount_upper = []
    
    for i, day in enumerate(unique_days):
        # Rain probability statistics
        pp_rain_mean.append(np.mean(pp_rain_probs[i]))
        pp_rain_lower.append(np.percentile(pp_rain_probs[i], 2.5))
        pp_rain_upper.append(np.percentile(pp_rain_probs[i], 97.5))
        
        # Rainfall amount statistics
        pp_amount_mean.append(np.mean(pp_rainfall_amounts[i]))
        pp_amount_lower.append(np.percentile(pp_rainfall_amounts[i], 2.5))
        pp_amount_upper.append(np.percentile(pp_rainfall_amounts[i], 97.5))
    
    # Convert to numpy arrays
    pp_rain_mean = np.array(pp_rain_mean)
    pp_rain_lower = np.array(pp_rain_lower)
    pp_rain_upper = np.array(pp_rain_upper)
    pp_amount_mean = np.array(pp_amount_mean)
    pp_amount_lower = np.array(pp_amount_lower)
    pp_amount_upper = np.array(pp_amount_upper)
    
    # Calculate observed statistics
    observed_rain_probs = []
    observed_rainfall_amounts = []
    
    for day in unique_days:
        day_mask = days_of_year == day
        day_data = data[day_mask]
        
        # Observed rain probability
        rain_days = (day_data['PRCP'] > 0).sum()
        total_days = len(day_data)
        observed_rain_probs.append(rain_days / total_days)
        
        # Observed rainfall amount (only rainy days)
        rainy_data = day_data[day_data['PRCP'] > 0]
        if len(rainy_data) > 0:
            observed_rainfall_amounts.append(rainy_data['PRCP'].mean())
        else:
            observed_rainfall_amounts.append(0)
    
    observed_rain_probs = np.array(observed_rain_probs)
    observed_rainfall_amounts = np.array(observed_rainfall_amounts)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Convert unique days to dates for proper x-axis
    unique_dates = _day_of_year_to_dates(unique_days)
    
    # 1. Rain probability comparison
    ax1 = axes[0, 0]
    ax1.plot(unique_dates, pp_rain_mean, 'b-', linewidth=2, label='Posterior Predictive Mean')
    ax1.fill_between(unique_dates, pp_rain_lower, pp_rain_upper, alpha=0.3, color='blue', label='95% CI')
    ax1.scatter(unique_dates, observed_rain_probs, color='red', alpha=0.6, s=20, label='Observed')
    ax1.set_ylabel('Rain Probability')
    ax1.set_title('Rain Probability: Observed vs Posterior Predictive')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis with month labels
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    # 2. Rainfall amount comparison
    ax2 = axes[0, 1]
    ax2.plot(unique_dates, pp_amount_mean, 'g-', linewidth=2, label='Posterior Predictive Mean')
    ax2.fill_between(unique_dates, pp_amount_lower, pp_amount_upper, alpha=0.3, color='green', label='95% CI')
    ax2.scatter(unique_dates, observed_rainfall_amounts, color='red', alpha=0.6, s=20, label='Observed')
    ax2.set_ylabel('Expected Rainfall (mm)')
    ax2.set_title('Rainfall Amount: Observed vs Posterior Predictive')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis with month labels
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    # 3. Residuals for rain probability
    ax3 = axes[1, 0]
    rain_residuals = observed_rain_probs - pp_rain_mean
    ax3.scatter(unique_dates, rain_residuals, alpha=0.6, s=20)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Residuals (Observed - Predicted)')
    ax3.set_title('Rain Probability Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis with month labels
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    # 4. Residuals for rainfall amount
    ax4 = axes[1, 1]
    amount_residuals = observed_rainfall_amounts - pp_amount_mean
    ax4.scatter(unique_dates, amount_residuals, alpha=0.6, s=20)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_ylabel('Residuals (Observed - Predicted)')
    ax4.set_title('Rainfall Amount Residuals')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis with month labels
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax4.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("POSTERIOR PREDICTIVE CHECK SUMMARY")
    print("=" * 50)
    print(f"Rain Probability RMSE: {np.sqrt(np.mean(rain_residuals**2)):.4f}")
    print(f"Rainfall Amount RMSE: {np.sqrt(np.mean(amount_residuals**2)):.4f}")
    print(f"Rain Probability MAE: {np.mean(np.abs(rain_residuals)):.4f}")
    print(f"Rainfall Amount MAE: {np.mean(np.abs(amount_residuals)):.4f}")
    
    return {
        'unique_days': unique_days,
        'pp_rain_mean': pp_rain_mean,
        'pp_rain_lower': pp_rain_lower,
        'pp_rain_upper': pp_rain_upper,
        'pp_amount_mean': pp_amount_mean,
        'pp_amount_lower': pp_amount_lower,
        'pp_amount_upper': pp_amount_upper,
        'observed_rain_probs': observed_rain_probs,
        'observed_rainfall_amounts': observed_rainfall_amounts,
        'rain_residuals': rain_residuals,
        'amount_residuals': amount_residuals
    }
