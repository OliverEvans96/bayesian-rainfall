import pymc as pm
import numpy as np
import pandas as pd


def load_data(filepath):
    """Load rainfall data and extract day of year."""
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['day_of_year'] = df['DATE'].dt.dayofyear
    return df[['DATE', 'PRCP', 'day_of_year']].dropna()


def create_rainfall_model(data, n_harmonics=5):
    """Bayesian model for rainfall based on day of year using multiple harmonics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing 'day_of_year' and 'PRCP' columns
    n_harmonics : int, default=3
        Number of harmonic pairs (sin/cos) to include. 
        n_harmonics=1 gives annual cycle, n_harmonics=2 adds semi-annual, etc.
    """
    all_days_of_year = data['day_of_year'].values  # 1-365
    all_rainfall = data['PRCP'].values

    # Create rain indicator
    all_is_rainy = (all_rainfall > 0).astype(int)
    
    # Get data for rainy days only
    all_rainy_mask = all_rainfall > 0
    all_rainy_rainfall = all_rainfall[all_rainy_mask]
    all_rainy_days_of_year = all_days_of_year[all_rainy_mask]
    
    with pm.Model() as model:
        # Rain probability (binomial) - multiple harmonics
        # Coefficients for each harmonic pair
        a_rain = pm.Normal("a_rain", mu=0, sigma=1.0, shape=n_harmonics)  # sin coefficients
        b_rain = pm.Normal("b_rain", mu=0, sigma=1.0, shape=n_harmonics)  # cos coefficients
        c_rain = pm.Normal("c_rain", mu=0, sigma=1.0)  # intercept
        
        # Create harmonic features for all days (vectorized)
        # Shape: (n_harmonics, n_days)
        h_values = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
        day_values = all_days_of_year[None, :]  # Shape: (1, n_days)
        
        # Vectorized harmonic calculations
        sin_features = pm.math.sin(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
        cos_features = pm.math.cos(2 * h_values * np.pi * day_values / 365.25)  # Shape: (n_harmonics, n_days)
        
        # Logit of rain probability using all harmonics (vectorized)
        # a_rain: (n_harmonics,), sin_features: (n_harmonics, n_days)
        # Result: (n_days,) - sum over harmonics dimension
        logit_p = c_rain + pm.math.sum(a_rain[:, None] * sin_features + b_rain[:, None] * cos_features, axis=0)
        
        p_rain = pm.Deterministic("p_rain", pm.math.sigmoid(logit_p))
        
        # Rain indicator (binomial)
        rain_indicator = pm.Bernoulli("rain_indicator", p=p_rain, observed=all_is_rainy)
        
        # Rainfall amount when it rains (gamma) - multiple harmonics
        # Coefficients for each harmonic pair
        a_amount = pm.Normal("a_amount", mu=0, sigma=1.0, shape=n_harmonics)  # sin coefficients
        b_amount = pm.Normal("b_amount", mu=0, sigma=1.0, shape=n_harmonics)  # cos coefficients
        c_amount = pm.Normal("c_amount", mu=1.0, sigma=1.0)  # intercept
        
        # Create harmonic features for rainy days only (vectorized)
        # Shape: (n_harmonics, n_rainy_days)
        h_values_rainy = np.arange(1, n_harmonics + 1)[:, None]  # Shape: (n_harmonics, 1)
        rainy_day_values = all_rainy_days_of_year[None, :]  # Shape: (1, n_rainy_days)
        
        # Vectorized harmonic calculations for rainy days
        rainy_sin_features = pm.math.sin(2 * h_values_rainy * np.pi * rainy_day_values / 365.25)  # Shape: (n_harmonics, n_rainy_days)
        rainy_cos_features = pm.math.cos(2 * h_values_rainy * np.pi * rainy_day_values / 365.25)  # Shape: (n_harmonics, n_rainy_days)
        
        # Expected rainfall amount using all harmonics (vectorized)
        # a_amount: (n_harmonics,), rainy_sin_features: (n_harmonics, n_rainy_days)
        # Result: (n_rainy_days,) - sum over harmonics dimension
        log_mu_amount = c_amount + pm.math.sum(a_amount[:, None] * rainy_sin_features + b_amount[:, None] * rainy_cos_features, axis=0)
        
        mu_amount = pm.math.exp(log_mu_amount)
        
        # Shape parameter for gamma
        alpha_amount = pm.Gamma("alpha_amount", alpha=2.0, beta=1.0)
        
        # Rainfall amount (only for days when it rains)
        rainfall_amount = pm.Gamma("rainfall_amount", 
                                 alpha=alpha_amount, 
                                 beta=alpha_amount/mu_amount, 
                                 observed=all_rainy_rainfall)
    
    return model


def sample_model(model, draws=1000, tune=1000):
    """Sample from the model."""
    with model:
        trace = pm.sample(draws=draws, tune=tune, random_seed=42)
    return trace


def create_model(n_harmonics=3):
    """Create the rainfall model with multiple harmonics.
    
    Parameters:
    -----------
    n_harmonics : int, default=3
        Number of harmonic pairs (sin/cos) to include
    """
    data = load_data("data/noaa_historical_weather_eugene_or_2019-2024.csv")
    model = create_rainfall_model(data, n_harmonics=n_harmonics)
    return model, data