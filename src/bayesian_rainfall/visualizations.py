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
from .analysis import _evaluate_model_for_day


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


def plot_combined_predictions(trace, data, ci_level=0.95, figsize=(14, 10)):
    """
    Plot combined rain probability and amount predictions across the year.
    
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
    # Calculate percentiles
    lower_percentile = 50 - (ci_level * 50)
    upper_percentile = 50 + (ci_level * 50)
    
    days_of_year = np.arange(1, 366)
    
    # Calculate rain probabilities and amounts for each day using the helper function
    all_rain_probs = []
    all_expected_amounts = []
    all_alpha_amounts = []
    
    for day in days_of_year:
        rain_probs, expected_amounts, alpha_amounts = _evaluate_model_for_day(trace, day)
        all_rain_probs.append(rain_probs)
        all_expected_amounts.append(expected_amounts)
        all_alpha_amounts.append(alpha_amounts)
    
    # Convert to arrays
    rain_probs = np.array(all_rain_probs)  # Shape: (365, n_samples)
    expected_amounts = np.array(all_expected_amounts)  # Shape: (365, n_samples)
    alpha_amounts = np.array(all_alpha_amounts)  # Shape: (365, n_samples)
    
    # Calculate statistics for expected probability
    rain_prob_mean = np.mean(rain_probs, axis=1)
    rain_prob_lower = np.percentile(rain_probs, lower_percentile, axis=1)
    rain_prob_upper = np.percentile(rain_probs, upper_percentile, axis=1)
    
    # Generate posterior predictive samples for rain indicators
    n_samples = 1000
    posterior_predictive_rain = []
    for i in range(min(n_samples, rain_probs.shape[1])):
        rain_indicators = np.random.binomial(1, rain_probs[:, i])
        posterior_predictive_rain.append(rain_indicators)
    
    posterior_predictive_rain = np.array(posterior_predictive_rain)
    pp_rain_mean = np.mean(posterior_predictive_rain, axis=0)
    pp_rain_lower = np.percentile(posterior_predictive_rain, lower_percentile, axis=0)
    pp_rain_upper = np.percentile(posterior_predictive_rain, upper_percentile, axis=0)
    
    # Calculate rainfall amounts
    rainfall_mean = np.mean(expected_amounts, axis=1)
    rainfall_lower = np.percentile(expected_amounts, 2.5, axis=1)
    rainfall_upper = np.percentile(expected_amounts, 97.5, axis=1)
    
    # Generate posterior predictive samples for rainfall amounts
    posterior_predictive_amounts = []
    for i in range(min(n_samples, rain_probs.shape[1])):
        rain_indicators = np.random.binomial(1, rain_probs[:, i])
        rainfall = np.zeros(len(days_of_year))
        rainy_mask = rain_indicators == 1
        if np.sum(rainy_mask) > 0:
            rainfall[rainy_mask] = np.random.gamma(alpha_amounts[rainy_mask, i], 
                                                 expected_amounts[rainy_mask, i] / alpha_amounts[rainy_mask, i])
        
        posterior_predictive_amounts.append(rainfall)
    
    posterior_predictive_amounts = np.array(posterior_predictive_amounts)
    pp_amounts_mean = np.mean(posterior_predictive_amounts, axis=0)
    pp_amounts_lower = np.percentile(posterior_predictive_amounts, 2.5, axis=0)
    pp_amounts_upper = np.percentile(posterior_predictive_amounts, 97.5, axis=0)
    
    # Get observed data
    observed_rain_prob = data.groupby('day_of_year')['PRCP'].apply(lambda x: (x > 0).mean()).values
    observed_days = data.groupby('day_of_year')['PRCP'].apply(lambda x: (x > 0).mean()).index.values
    rainy_data = data[data['PRCP'] > 0]
    observed_rainfall = rainy_data.groupby('day_of_year')['PRCP'].mean().values
    observed_rainy_days = rainy_data.groupby('day_of_year')['PRCP'].mean().index.values
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Rain probability plot
    ax1.fill_between(days_of_year, rain_prob_lower, rain_prob_upper, alpha=0.2, color='blue', 
                    label=f'{int(ci_level*100)}% CI (Expected Probability)')
    ax1.plot(days_of_year, rain_prob_mean, 'b-', linewidth=2, label='Expected Probability')
    ax1.fill_between(days_of_year, pp_rain_lower, pp_rain_upper, alpha=0.3, color='green', 
                    label=f'{int(ci_level*100)}% CI (Posterior Predictive)')
    ax1.plot(days_of_year, pp_rain_mean, 'g--', linewidth=2, label='Posterior Predictive Mean')
    ax1.scatter(observed_days, observed_rain_prob, alpha=0.6, color='red', s=20, label='Observed')
    ax1.set_xlabel('Day of Year')
    ax1.set_ylabel('Rain Probability')
    ax1.set_title('Rain Probability Predictions Across the Year')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rainfall amount plot
    ax2.fill_between(days_of_year, rainfall_lower, rainfall_upper, alpha=0.2, color='green', 
                    label='95% CI (Expected Amount)')
    ax2.plot(days_of_year, rainfall_mean, 'g-', linewidth=2, label='Expected Amount')
    ax2.fill_between(days_of_year, pp_amounts_lower, pp_amounts_upper, alpha=0.3, color='orange', 
                    label='95% CI (Posterior Predictive)')
    ax2.plot(days_of_year, pp_amounts_mean, 'orange', linestyle='--', linewidth=2, label='Posterior Predictive Mean')
    ax2.scatter(observed_rainy_days, observed_rainfall, alpha=0.6, color='red', s=20, label='Observed (Rainy Days)')
    ax2.set_xlabel('Day of Year')
    ax2.set_ylabel('Expected Rainfall Amount (mm)')
    ax2.set_title('Rainfall Amount Predictions for Rainy Days Across the Year')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_posterior_predictive_checks(trace, model, data, n_samples=100, figsize=(15, 10)):
    """
    Plot posterior predictive checks comparing observed vs predicted data.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from sampling
    model : pymc.Model
        PyMC model
    data : pandas.DataFrame
        Weather data with columns 'PRCP' and 'day_of_year'
    n_samples : int, optional
        Number of posterior samples to use for predictions
    figsize : tuple, optional
        Figure size (width, height)
    """
    # Generate posterior predictive samples
    with model:
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['rain_indicator'])
    
    predicted_rain_indicator = posterior_predictive.posterior_predictive.rain_indicator.values
    
    # Generate full rainfall predictions
    full_rainfall_predictions = []
    
    for i in range(min(n_samples, predicted_rain_indicator.shape[0] * predicted_rain_indicator.shape[1])):
        chain_idx = i // predicted_rain_indicator.shape[1]
        sample_idx = i % predicted_rain_indicator.shape[1]
        
        # Get parameters for this sample
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
    
    axes[1, 0].plot(first_100_days['PRCP'].values, 'b-', label='Observed', alpha=0.7)
    axes[1, 0].plot(np.mean(first_100_pred, axis=0), 'r-', label='Predicted Mean', alpha=0.7)
    axes[1, 0].fill_between(range(100), 
                           np.percentile(first_100_pred, 2.5, axis=0),
                           np.percentile(first_100_pred, 97.5, axis=0),
                           alpha=0.3, color='red', label='95% CI')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Rainfall (mm)')
    axes[1, 0].set_title('Time Series: First 100 Days')
    axes[1, 0].legend()
    
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
    
    for i, (day, day_name) in enumerate(zip(selected_days, day_names)):
        # Get observed data for this day of year across all years
        day_data = data[data['day_of_year'] == day]['PRCP'].values
        
        # Get predicted data for this day of year using the helper function
        rain_probs, expected_amounts, alpha_amounts = _evaluate_model_for_day(trace, day)
        
        # Sample predictions for this specific day
        n_samples = min(100, len(rain_probs))
        day_predictions = []
        
        for j in range(n_samples):
            # Sample rain indicator
            rain_indicator = np.random.binomial(1, rain_probs[j])
            
            # Sample rainfall amount if it rains
            if rain_indicator == 1:
                rainfall = np.random.gamma(alpha_amounts[j], expected_amounts[j] / alpha_amounts[j])
            else:
                rainfall = 0
            
            day_predictions.append(rainfall)
        
        day_predictions = np.array(day_predictions)
        
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
    
    # Ensure we have probabilities for all 52 weeks
    observed_weekly_probs_full = np.zeros(52)
    for week in range(1, 53):
        if week in observed_weekly_probs.index:
            observed_weekly_probs_full[week-1] = observed_weekly_probs[week]
        else:
            observed_weekly_probs_full[week-1] = 0  # No data for this week
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]//2))
    
    # Weekly rain probability with confidence intervals
    ax.fill_between(weeks, lower_ci, upper_ci, alpha=0.2, color='blue', 
                    label='95% CI (Expected Probability)')
    ax.plot(weeks, mean_probs, 'b-', linewidth=2, label='Expected Probability')
    ax.fill_between(weeks, pp_lower_ci, pp_upper_ci, alpha=0.3, color='green', 
                    label='95% CI (Posterior Predictive)')
    ax.plot(weeks, pp_mean, 'g--', linewidth=2, label='Posterior Predictive Mean')
    ax.scatter(weeks, observed_weekly_probs_full, color='red', s=30, alpha=0.7, 
               label='Observed Probability', zorder=5)
    ax.set_xlabel('Week of Year')
    ax.set_ylabel('Probability of Any Rain')
    ax.set_title('Weekly Rain Probability Throughout the Year', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add month labels on x-axis
    month_positions = [1, 5, 9, 13, 17, 22, 26, 30, 35, 39, 43, 47, 52]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels)
    
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
