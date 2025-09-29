import pymc as pm
import numpy as np
import pandas as pd


def load_data(filepath):
    """Load rainfall data and extract day of year and year."""
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['year'] = df['DATE'].dt.year
    return df[['DATE', 'PRCP', 'day_of_year', 'year']].dropna()


def create_model(data, n_harmonics=5, include_trend=False, noncentered=True):
    """Bayesian hierarchical model for rainfall with year-to-year variation.
    
    This model adds year-specific random effects to capture year-to-year variation
    while maintaining the seasonal harmonic structure. Year effects are constrained
    to sum to zero to improve identifiability.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing 'day_of_year', 'year', and 'PRCP' columns
    n_harmonics : int, default=5
        Number of harmonic pairs (sin/cos) to include
    include_trend : bool, default=False
        Whether to include a linear trend component over years
    noncentered : bool, default=True
        Whether to use non-centered parameterization for year effects (helps with sampling)
        
    Notes:
    ------
    Year effects are constrained to sum to zero (sum(year_effects) = 0) to improve
    identifiability and reduce correlation between global intercepts and year effects.
    This ensures that the global intercept represents the "average year" effect.
    """
    all_days_of_year = data['day_of_year'].values  # 1-365
    all_rainfall = data['PRCP'].values
    all_years = data['year'].values

    # Create rain indicator
    all_is_rainy = (all_rainfall > 0).astype(int)
    
    # Get data for rainy days only
    all_rainy_mask = all_rainfall > 0
    all_rainy_rainfall = all_rainfall[all_rainy_mask]
    all_rainy_days_of_year = all_days_of_year[all_rainy_mask]
    all_rainy_years = all_years[all_rainy_mask]
    
    # Get unique years and create year indices
    unique_years = np.sort(data['year'].unique())
    year_to_idx = {year: idx for idx, year in enumerate(unique_years)}
    all_year_indices = np.array([year_to_idx[year] for year in all_years])
    all_rainy_year_indices = np.array([year_to_idx[year] for year in all_rainy_years])
    n_years = len(unique_years)
    
    with pm.Model() as model:
        # Add year coordinates for better trace handling
        model.add_coord("year", unique_years)
        
        # Rain probability (binomial) - multiple harmonics with year effects
        a_rain = pm.Normal("a_rain", mu=0, sigma=1.0, shape=n_harmonics)  # sin coefficients
        b_rain = pm.Normal("b_rain", mu=0, sigma=1.0, shape=n_harmonics)  # cos coefficients
        c_rain = pm.Normal("c_rain", mu=0, sigma=2.0)  # intercept
        
        # Year-specific random effects for rain probability
        year_rain_sigma = pm.HalfNormal("year_rain_sigma", sigma=0.5)
        
        if noncentered:
            # Non-centered parameterization with sum-to-zero constraint
            year_rain_raw = pm.Normal("year_rain_raw", mu=0, sigma=1, shape=n_years, dims="year")
            # Apply sum-to-zero constraint: subtract the mean
            year_rain_effects_raw = year_rain_sigma * year_rain_raw
            year_rain_effects = pm.Deterministic("year_rain_effects", 
                                               year_rain_effects_raw - pm.math.mean(year_rain_effects_raw), dims="year")
        else:
            # Centered parameterization with sum-to-zero constraint
            year_rain_effects_raw = pm.Normal("year_rain_effects_raw", mu=0, sigma=year_rain_sigma, shape=n_years, dims="year")
            year_rain_effects = pm.Deterministic("year_rain_effects", 
                                               year_rain_effects_raw - pm.math.mean(year_rain_effects_raw), dims="year")
        
        # Optional linear trend over years
        if include_trend:
            trend_rain = pm.Normal("trend_rain", mu=0, sigma=0.1)
            trend_rain_contribution = trend_rain * (all_years - all_years.mean())
        else:
            trend_rain_contribution = 0
        
        # Create harmonic features for all days (vectorized)
        h_values = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
        day_values = all_days_of_year[None, :]  # Shape: (1, n_days)
        
        # Vectorized harmonic calculations
        sin_features = pm.math.sin(2 * h_values * np.pi * day_values / 365.25)
        cos_features = pm.math.cos(2 * h_values * np.pi * day_values / 365.25)
        
        # Logit of rain probability using all harmonics + year effects + trend
        logit_p = (c_rain + 
                  pm.math.sum(a_rain[:, None] * sin_features + b_rain[:, None] * cos_features, axis=0) +
                  year_rain_effects[all_year_indices] +
                  trend_rain_contribution)
        
        p_rain = pm.Deterministic("p_rain", pm.math.sigmoid(logit_p))
        
        # Rain indicator (binomial)
        rain_indicator = pm.Bernoulli("rain_indicator", p=p_rain, observed=all_is_rainy)
        
        # Rainfall amount when it rains (gamma) - multiple harmonics with year effects
        a_amount = pm.Normal("a_amount", mu=0, sigma=1.0, shape=n_harmonics)  # sin coefficients
        b_amount = pm.Normal("b_amount", mu=0, sigma=1.0, shape=n_harmonics)  # cos coefficients
        c_amount = pm.Normal("c_amount", mu=1.0, sigma=1.0)  # intercept
        
        # Year-specific random effects for rainfall amount
        year_amount_sigma = pm.HalfNormal("year_amount_sigma", sigma=0.3)
        
        if noncentered:
            # Non-centered parameterization with sum-to-zero constraint
            year_amount_raw = pm.Normal("year_amount_raw", mu=0, sigma=1, shape=n_years, dims="year")
            # Apply sum-to-zero constraint: subtract the mean
            year_amount_effects_raw = year_amount_sigma * year_amount_raw
            year_amount_effects = pm.Deterministic("year_amount_effects", 
                                                 year_amount_effects_raw - pm.math.mean(year_amount_effects_raw), dims="year")
        else:
            # Centered parameterization with sum-to-zero constraint
            year_amount_effects_raw = pm.Normal("year_amount_effects_raw", mu=0, sigma=year_amount_sigma, shape=n_years, dims="year")
            year_amount_effects = pm.Deterministic("year_amount_effects", 
                                                 year_amount_effects_raw - pm.math.mean(year_amount_effects_raw), dims="year")
        
        # Optional linear trend over years for amount
        if include_trend:
            trend_amount = pm.Normal("trend_amount", mu=0, sigma=0.1)
            trend_amount_contribution = trend_amount * (all_rainy_years - all_rainy_years.mean())
        else:
            trend_amount_contribution = 0
        
        # Create harmonic features for rainy days only (vectorized)
        h_values_rainy = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
        rainy_day_values = all_rainy_days_of_year[None, :]  # Shape: (1, n_rainy_days)
        
        # Vectorized harmonic calculations for rainy days
        rainy_sin_features = pm.math.sin(2 * h_values_rainy * np.pi * rainy_day_values / 365.25)
        rainy_cos_features = pm.math.cos(2 * h_values_rainy * np.pi * rainy_day_values / 365.25)
        
        # Expected rainfall amount for ALL days (not just rainy days)
        # This allows us to use mu_amount for any day of the year
        log_mu_amount_all = (c_amount + 
                            pm.math.sum(a_amount[:, None] * sin_features + b_amount[:, None] * cos_features, axis=0) +
                            year_amount_effects[all_year_indices] +
                            trend_amount_contribution)
        
        mu_amount_all = pm.Deterministic("mu_amount", pm.math.exp(log_mu_amount_all))
        
        # Expected rainfall amount for rainy days only (for the observed data)
        log_mu_amount_rainy = (c_amount + 
                              pm.math.sum(a_amount[:, None] * rainy_sin_features + b_amount[:, None] * rainy_cos_features, axis=0) +
                              year_amount_effects[all_rainy_year_indices] +
                              trend_amount_contribution)
        
        mu_amount_rainy = pm.math.exp(log_mu_amount_rainy)
        
        # Shape parameter for gamma
        alpha_amount = pm.Gamma("alpha_amount", alpha=2.0, beta=1.0)  # Mean=2.0
        
        # Rainfall amount (only for days when it rains)
        rainfall_amount = pm.Gamma("rainfall_amount", 
                                 alpha=alpha_amount, 
                                 beta=alpha_amount/mu_amount_rainy, 
                                 observed=all_rainy_rainfall)
        
        # Store year information for later use
        model.year_to_idx = year_to_idx
        model.unique_years = unique_years
    
    return model


def sample_model(model, draws=1000, tune=1000, **kwargs):
    """Sample from the model with improved parameters for hierarchical models.
    
    Parameters:
    -----------
    model : pm.Model
        PyMC model to sample from
    draws : int, default=1000
        Number of samples to draw
    tune : int, default=2000
        Number of tuning samples (increased for hierarchical models)
    target_accept : float, default=0.95
        Target acceptance rate (higher for hierarchical models)
    max_treedepth : int, default=12
        Maximum tree depth for NUTS
    """
    with model:
        # Use more conservative sampling parameters for hierarchical models
        trace = pm.sample(
            draws=draws, 
            tune=tune, 
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
            **kwargs
        )
    return trace




def check_sampling_quality(trace):
    """Check sampling quality and print diagnostics.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace to check
        
    Returns:
    --------
    dict : Sampling diagnostics
    """
    # Check for divergences
    n_divergences = trace.sample_stats.diverging.sum().item()
    n_samples = trace.sample_stats.diverging.size
    divergence_rate = n_divergences / n_samples
    
    print("SAMPLING QUALITY DIAGNOSTICS")
    print("=" * 40)
    print(f"Total samples: {n_samples}")
    print(f"Divergences: {n_divergences} ({divergence_rate:.1%})")
    
    # Simple quality assessment based on divergences
    quality_issues = []
    if divergence_rate > 0.01:
        quality_issues.append(f"High divergence rate: {divergence_rate:.1%}")
    
    if quality_issues:
        print("\n⚠️  SAMPLING ISSUES DETECTED:")
        for issue in quality_issues:
            print(f"  - {issue}")
        print("\n💡 RECOMMENDATIONS:")
        if divergence_rate > 0.01:
            print("  - Increase tune samples (try tune=5000)")
            print("  - Increase target_accept (try 0.98)")
            print("  - Try non-centered parameterization")
            print("  - Consider mathematical modifications for identifiability")
    else:
        print("\n✅ SAMPLING QUALITY LOOKS GOOD!")
    
    return {
        'n_divergences': n_divergences,
        'divergence_rate': divergence_rate,
        'quality_issues': quality_issues
    }