from bayesian_rainfall.model import load_data, create_model

def test_data_loading():
    """Test that data loading works correctly."""
    data = load_data("data/noaa_historical_weather_eugene_or_2019-2024.csv")
    
    # Check that we have the expected columns
    expected_cols = ['PRCP', 'day_of_year']
    for col in expected_cols:
        assert col in data.columns, f"Missing column: {col}"
    
    # Check that we have data
    assert len(data) > 0, "No data loaded"
    
    # Check that rainfall data is non-negative
    assert (data['PRCP'] >= 0).all(), "Rainfall data contains negative values"


def test_model_creation():
    """Test that the model can be created without errors."""
    data = load_data("data/noaa_historical_weather_eugene_or_2019-2024.csv")
    model = create_model(data)
    
    # Check that the model has the expected variables
    expected_vars = ['a_rain', 'b_rain', 'c_rain', 'p_rain', 'rain_indicator', 
                    'a_amount', 'b_amount', 'c_amount', 'alpha_amount', 'rainfall_amount']
    
    for var in expected_vars:
        assert var in model.named_vars, f"Missing variable in model: {var}"
