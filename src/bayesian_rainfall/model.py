import pymc as pm
import numpy as np
import pandas as pd


def load_data(filepath):
    """Load rainfall data and extract day of year."""
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['day_of_year'] = df['DATE'].dt.dayofyear
    return df[['DATE', 'PRCP', 'day_of_year']].dropna()


def create_rainfall_model(data):
    """Simple Bayesian model for rainfall based on day of year."""
    all_days_of_year = data['day_of_year'].values  # 1-365
    all_rainfall = data['PRCP'].values

    # Create rain indicator
    all_is_rainy = (all_rainfall > 0).astype(int)
    
    # Get data for rainy days only
    all_rainy_mask = all_rainfall > 0
    all_rainy_rainfall = all_rainfall[all_rainy_mask]
    all_rainy_days_of_year = all_days_of_year[all_rainy_mask]
    
    with pm.Model() as model:
        # Rain probability (binomial)
        a_rain = pm.Normal("a_rain", mu=0, sigma=2.0)  # sin coefficient for rain prob
        b_rain = pm.Normal("b_rain", mu=0, sigma=2.0)  # cos coefficient for rain prob
        c_rain = pm.Normal("c_rain", mu=0, sigma=1.0)  # intercept for rain prob
        
        # Derive cyclical encodings from day_of_year
        day_sin = pm.math.sin(2 * np.pi * all_days_of_year / 365.25)
        day_cos = pm.math.cos(2 * np.pi * all_days_of_year / 365.25)
        
        # Logit of rain probability
        logit_p = c_rain + a_rain * day_sin + b_rain * day_cos
        p_rain = pm.Deterministic("p_rain", pm.math.sigmoid(logit_p))
        
        # Rain indicator (binomial)
        rain_indicator = pm.Bernoulli("rain_indicator", p=p_rain, observed=all_is_rainy)
        
        # Rainfall amount when it rains (gamma)
        a_amount = pm.Normal("a_amount", mu=0, sigma=2.0)  # sin coefficient for amount
        b_amount = pm.Normal("b_amount", mu=0, sigma=2.0)  # cos coefficient for amount
        c_amount = pm.Normal("c_amount", mu=1.0, sigma=1.0)  # intercept for amount
        
        # Expected rainfall amount (only for rainy days)
        # Use the cyclical encodings for rainy days only
        rainy_day_sin = pm.math.sin(2 * np.pi * all_rainy_days_of_year / 365.25)
        rainy_day_cos = pm.math.cos(2 * np.pi * all_rainy_days_of_year / 365.25)
        mu_amount = pm.math.exp(c_amount + a_amount * rainy_day_sin + b_amount * rainy_day_cos)
        
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


def create_model():
    """Create the rainfall model."""
    data = load_data("data/noaa_historical_weather_eugene_or_2019-2024.csv")
    model = create_rainfall_model(data)
    return model, data