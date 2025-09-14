"""
Comprehensive tests for the analysis module.
"""

import pytest
import numpy as np
import pandas as pd
import arviz as az
from unittest.mock import Mock, patch
from datetime import datetime

from bayesian_rainfall.analysis import (
    _parse_date_input,
    print_model_summary,
    print_convergence_diagnostics,
    analyze_single_day,
    calculate_rainfall_interval_probability,
    calculate_any_rain_probability
)


class TestParseDateInput:
    """Test the _parse_date_input function."""
    
    def test_parse_day_of_year_int(self):
        """Test parsing day of year as integer."""
        # Test valid day of year
        day_of_year, day_name = _parse_date_input(15)
        assert day_of_year == 15
        assert day_name == "January 15"
        
        # Test day 365 (December 30 in 2024, which is a leap year)
        day_of_year, day_name = _parse_date_input(365)
        assert day_of_year == 365
        assert day_name == "December 30"  # 2024 is a leap year
        
        # Test day 1
        day_of_year, day_name = _parse_date_input(1)
        assert day_of_year == 1
        assert day_name == "January 01"
    
    def test_parse_day_of_year_int_invalid(self):
        """Test parsing invalid day of year integers."""
        with pytest.raises(ValueError, match="day_of_year must be between 1 and 365"):
            _parse_date_input(0)
        
        with pytest.raises(ValueError, match="day_of_year must be between 1 and 365"):
            _parse_date_input(366)
        
        with pytest.raises(ValueError, match="day_of_year must be between 1 and 365"):
            _parse_date_input(-1)
    
    def test_parse_date_string(self):
        """Test parsing date as string in MM/DD format."""
        # Test January 15
        day_of_year, day_name = _parse_date_input("01/15")
        assert day_of_year == 15
        assert day_name == "January 15"
        
        # Test December 31 (366th day in 2024 leap year)
        day_of_year, day_name = _parse_date_input("12/31")
        assert day_of_year == 366  # 2024 is a leap year
        assert day_name == "December 31"
        
        # Test February 29 (leap year)
        day_of_year, day_name = _parse_date_input("02/29")
        assert day_of_year == 60
        assert day_name == "February 29"
    
    def test_parse_date_string_invalid(self):
        """Test parsing invalid date strings."""
        with pytest.raises(ValueError, match="Invalid date format"):
            _parse_date_input("13/15")
        
        with pytest.raises(ValueError, match="Invalid date format"):
            _parse_date_input("01/32")
        
        with pytest.raises(ValueError, match="Invalid date format"):
            _parse_date_input("invalid")
        
        # Test that "1/15" actually works (it's a valid date format)
        day_of_year, day_name = _parse_date_input("1/15")
        assert day_of_year == 15
        assert day_name == "January 15"
    
    def test_parse_date_tuple(self):
        """Test parsing date as tuple (month, day)."""
        # Test January 15
        day_of_year, day_name = _parse_date_input((1, 15))
        assert day_of_year == 15
        assert day_name == "January 15"
        
        # Test December 31 (366th day in 2024 leap year)
        day_of_year, day_name = _parse_date_input((12, 31))
        assert day_of_year == 366  # 2024 is a leap year
        assert day_name == "December 31"
        
        # Test February 29
        day_of_year, day_name = _parse_date_input((2, 29))
        assert day_of_year == 60
        assert day_name == "February 29"
    
    def test_parse_date_tuple_invalid(self):
        """Test parsing invalid date tuples."""
        with pytest.raises(ValueError, match="Month must be 1-12, day must be 1-31"):
            _parse_date_input((13, 15))
        
        with pytest.raises(ValueError, match="Month must be 1-12, day must be 1-31"):
            _parse_date_input((1, 32))
        
        with pytest.raises(ValueError, match="Month must be 1-12, day must be 1-31"):
            _parse_date_input((0, 15))
        
        with pytest.raises(ValueError, match="Month must be 1-12, day must be 1-31"):
            _parse_date_input((1, 0))
    
    def test_parse_date_list(self):
        """Test parsing date as list [month, day]."""
        day_of_year, day_name = _parse_date_input([1, 15])
        assert day_of_year == 15
        assert day_name == "January 15"
    
    def test_parse_date_invalid_type(self):
        """Test parsing invalid input types."""
        with pytest.raises(ValueError, match="Invalid date format"):
            _parse_date_input("not a date")
        
        with pytest.raises(ValueError, match="date_input must be int"):
            _parse_date_input(1.5)
        
        with pytest.raises(ValueError, match="date_input must be int"):
            _parse_date_input((1, 2, 3))  # Too many elements


class TestPrintModelSummary:
    """Test the print_model_summary function."""
    
    def create_mock_trace(self):
        """Create a mock trace for testing."""
        # Create mock posterior data
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 1000)),
            'b_rain': np.random.normal(0, 1, (2, 1000)),
            'c_rain': np.random.normal(0, 1, (2, 1000)),
            'a_amount': np.random.normal(0, 1, (2, 1000)),
            'b_amount': np.random.normal(0, 1, (2, 1000)),
            'c_amount': np.random.normal(1, 1, (2, 1000)),
            'alpha_amount': np.random.gamma(2, 1, (2, 1000))
        }
        
        # Create mock trace
        trace = Mock()
        trace.posterior = Mock()
        
        # Set up the posterior data properly
        for param, values in posterior_data.items():
            param_mock = Mock()
            param_mock.values = values
            param_mock.shape = values.shape
            setattr(trace.posterior, param, param_mock)
        
        # Set up __getitem__ to return the parameter mocks
        def getitem_side_effect(key):
            if hasattr(trace.posterior, key):
                return getattr(trace.posterior, key)
            return Mock(values=np.random.normal(0, 1, (2, 1000)))
        
        trace.posterior.__getitem__ = getitem_side_effect
        
        return trace
    
    def create_mock_data(self):
        """Create mock weather data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        np.random.seed(42)
        rainfall = np.random.exponential(0.1, len(dates))
        rainfall[rainfall < 0.05] = 0  # Make some days dry
        
        return pd.DataFrame({
            'DATE': dates,
            'PRCP': rainfall,
            'day_of_year': dates.dayofyear
        })
    
    @patch('builtins.print')
    @patch('bayesian_rainfall.analysis.az.rhat')
    def test_print_model_summary_basic(self, mock_rhat, mock_print):
        """Test basic functionality of print_model_summary."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        
        # Mock the rhat function to return a mock with values
        rhat_result = Mock()
        rhat_result.__getitem__ = Mock(side_effect=lambda x: Mock(values=np.array(1.01)))
        mock_rhat.return_value = rhat_result
        
        print_model_summary(trace, data)
        
        # Check that print was called multiple times
        assert mock_print.call_count > 10
        
        # Check for key sections in output
        printed_text = ' '.join([call[0][0] for call in mock_print.call_args_list])
        assert "MODEL SUMMARY" in printed_text
        assert "CONVERGENCE DIAGNOSTICS" in printed_text
        assert "MODEL FIT ANALYSIS" in printed_text
        assert "SEASONAL PATTERNS" in printed_text
    
    @patch('builtins.print')
    @patch('bayesian_rainfall.analysis.az.rhat')
    def test_print_model_summary_with_custom_params(self, mock_rhat, mock_print):
        """Test print_model_summary with custom parameter names."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        custom_params = ['a_rain', 'b_rain']
        
        # Mock the rhat function to return a mock with values
        rhat_result = Mock()
        rhat_result.__getitem__ = Mock(side_effect=lambda x: Mock(values=np.array(1.01)))
        mock_rhat.return_value = rhat_result
        
        print_model_summary(trace, data, param_names=custom_params)
        
        # Should still work with custom parameters
        assert mock_print.call_count > 5


class TestPrintConvergenceDiagnostics:
    """Test the print_convergence_diagnostics function."""
    
    def create_mock_trace(self):
        """Create a mock trace for testing."""
        trace = Mock()
        trace.posterior = Mock()
        
        # Mock arviz functions
        with patch('bayesian_rainfall.analysis.az.rhat') as mock_rhat, \
             patch('bayesian_rainfall.analysis.az.ess') as mock_ess:
            
            mock_rhat.return_value = Mock()
            mock_ess.return_value = Mock()
            
            return trace
    
    @patch('builtins.print')
    def test_print_convergence_diagnostics_basic(self, mock_print):
        """Test basic functionality of print_convergence_diagnostics."""
        trace = self.create_mock_trace()
        
        with patch('bayesian_rainfall.analysis.az.rhat') as mock_rhat, \
             patch('bayesian_rainfall.analysis.az.ess') as mock_ess:
            
            mock_rhat.return_value = Mock()
            mock_ess.return_value = Mock()
            
            print_convergence_diagnostics(trace)
            
            # Check that print was called
            assert mock_print.call_count > 0
            mock_rhat.assert_called_once()
            mock_ess.assert_called_once()
    
    @patch('builtins.print')
    def test_print_convergence_diagnostics_custom_params(self, mock_print):
        """Test print_convergence_diagnostics with custom parameters."""
        trace = self.create_mock_trace()
        custom_params = ['a_rain', 'b_rain']
        
        with patch('bayesian_rainfall.analysis.az.rhat') as mock_rhat, \
             patch('bayesian_rainfall.analysis.az.ess') as mock_ess:
            
            mock_rhat.return_value = Mock()
            mock_ess.return_value = Mock()
            
            print_convergence_diagnostics(trace, param_names=custom_params)
            
            # Should still work with custom parameters
            assert mock_print.call_count > 0


class TestAnalyzeSingleDay:
    """Test the analyze_single_day function."""
    
    def create_mock_trace(self):
        """Create a mock trace for testing."""
        # Create mock posterior data
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 100)),
            'b_rain': np.random.normal(0, 1, (2, 100)),
            'c_rain': np.random.normal(0, 1, (2, 100)),
            'a_amount': np.random.normal(0, 1, (2, 100)),
            'b_amount': np.random.normal(0, 1, (2, 100)),
            'c_amount': np.random.normal(1, 1, (2, 100)),
            'alpha_amount': np.random.gamma(2, 1, (2, 100))
        }
        
        # Create mock trace
        trace = Mock()
        trace.posterior = Mock()
        
        # Set up the posterior data properly
        for param, values in posterior_data.items():
            param_mock = Mock()
            param_mock.values = values
            param_mock.shape = values.shape
            setattr(trace.posterior, param, param_mock)
        
        # Set up __getitem__ to return the parameter mocks
        def getitem_side_effect(key):
            if hasattr(trace.posterior, key):
                return getattr(trace.posterior, key)
            return Mock(values=np.random.normal(0, 1, (2, 100)))
        
        trace.posterior.__getitem__ = getitem_side_effect
        
        return trace
    
    def create_mock_data(self):
        """Create mock weather data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        np.random.seed(42)
        rainfall = np.random.exponential(0.1, len(dates))
        rainfall[rainfall < 0.05] = 0  # Make some days dry
        
        return pd.DataFrame({
            'DATE': dates,
            'PRCP': rainfall,
            'day_of_year': dates.dayofyear
        })
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.print')
    def test_analyze_single_day_int_input(self, mock_print, mock_show):
        """Test analyze_single_day with integer day input."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        
        result = analyze_single_day(trace, data, 15, show_plots=True)
        
        # Check that result is returned
        assert result is not None
        assert 'day_of_year' in result
        assert 'day_name' in result
        assert 'observed' in result
        assert 'predicted' in result
        assert 'performance' in result
        
        # Check day information
        assert result['day_of_year'] == 15
        assert result['day_name'] == "January 15"
        
        # Check observed data structure
        assert 'data' in result['observed']
        assert 'mean' in result['observed']
        assert 'std' in result['observed']
        assert 'rain_frequency' in result['observed']
        
        # Check predicted data structure
        assert 'samples' in result['predicted']
        assert 'mean' in result['predicted']
        assert 'ci_95' in result['predicted']
        assert 'rain_probability_mean' in result['predicted']
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.print')
    def test_analyze_single_day_string_input(self, mock_print, mock_show):
        """Test analyze_single_day with string date input."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        
        result = analyze_single_day(trace, data, "01/15", show_plots=True)
        
        assert result is not None
        assert result['day_of_year'] == 15
        assert result['day_name'] == "January 15"
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.print')
    def test_analyze_single_day_tuple_input(self, mock_print, mock_show):
        """Test analyze_single_day with tuple date input."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        
        result = analyze_single_day(trace, data, (1, 15), show_plots=True)
        
        assert result is not None
        assert result['day_of_year'] == 15
        assert result['day_name'] == "January 15"
    
    @patch('builtins.print')
    def test_analyze_single_day_no_plots(self, mock_print):
        """Test analyze_single_day without plots."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        
        result = analyze_single_day(trace, data, 15, show_plots=False)
        
        assert result is not None
        assert result['day_of_year'] == 15
    
    @patch('builtins.print')
    def test_analyze_single_day_no_data(self, mock_print):
        """Test analyze_single_day when no data exists for the day."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        
        # Use a day that doesn't exist in the data (but is valid)
        # Create data that only has days 1-100
        limited_data = data[data['day_of_year'] <= 100]
        result = analyze_single_day(trace, limited_data, 200, show_plots=False)
        
        # Should return None because there's no data for day 200 in our test data
        assert result is None
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.print')
    def test_analyze_single_day_custom_figsize(self, mock_print, mock_show):
        """Test analyze_single_day with custom figure size."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        
        result = analyze_single_day(trace, data, 15, show_plots=True, figsize=(20, 15))
        
        assert result is not None


class TestCalculateRainfallIntervalProbability:
    """Test the calculate_rainfall_interval_probability function."""
    
    def create_mock_trace(self):
        """Create a mock trace for testing."""
        # Create mock posterior data
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 100)),
            'b_rain': np.random.normal(0, 1, (2, 100)),
            'c_rain': np.random.normal(0, 1, (2, 100)),
            'a_amount': np.random.normal(0, 1, (2, 100)),
            'b_amount': np.random.normal(0, 1, (2, 100)),
            'c_amount': np.random.normal(1, 1, (2, 100)),
            'alpha_amount': np.random.gamma(2, 1, (2, 100))
        }
        
        # Create mock trace
        trace = Mock()
        trace.posterior = Mock()
        
        # Set up the posterior data properly
        for param, values in posterior_data.items():
            param_mock = Mock()
            param_mock.values = values
            param_mock.shape = values.shape
            setattr(trace.posterior, param, param_mock)
        
        # Set up __getitem__ to return the parameter mocks
        def getitem_side_effect(key):
            if hasattr(trace.posterior, key):
                return getattr(trace.posterior, key)
            return Mock(values=np.random.normal(0, 1, (2, 100)))
        
        trace.posterior.__getitem__ = getitem_side_effect
        
        return trace
    
    @patch('builtins.print')
    def test_calculate_rainfall_interval_probability_any_rain(self, mock_print):
        """Test calculating probability of any rainfall."""
        trace = self.create_mock_trace()
        
        result = calculate_rainfall_interval_probability(trace, 15)
        
        assert result is not None
        assert 'day_of_year' in result
        assert 'day_name' in result
        assert 'probability' in result
        assert 'samples' in result
        assert result['day_of_year'] == 15
        assert result['day_name'] == "January 15"
        assert 0 <= result['probability'] <= 1
    
    @patch('builtins.print')
    def test_calculate_rainfall_interval_probability_upper_bound(self, mock_print):
        """Test calculating probability with upper bound only."""
        trace = self.create_mock_trace()
        
        result = calculate_rainfall_interval_probability(trace, 15, interval_max=0.5)
        
        assert result is not None
        assert result['interval_max'] == 0.5
        assert result['interval_min'] is None
        assert 0 <= result['probability'] <= 1
    
    @patch('builtins.print')
    def test_calculate_rainfall_interval_probability_lower_bound(self, mock_print):
        """Test calculating probability with lower bound only."""
        trace = self.create_mock_trace()
        
        result = calculate_rainfall_interval_probability(trace, 15, interval_min=0.1)
        
        assert result is not None
        assert result['interval_min'] == 0.1
        assert result['interval_max'] is None
        assert 0 <= result['probability'] <= 1
    
    @patch('builtins.print')
    def test_calculate_rainfall_interval_probability_both_bounds(self, mock_print):
        """Test calculating probability with both bounds."""
        trace = self.create_mock_trace()
        
        result = calculate_rainfall_interval_probability(trace, 15, 0.1, 0.5)
        
        assert result is not None
        assert result['interval_min'] == 0.1
        assert result['interval_max'] == 0.5
        assert 0 <= result['probability'] <= 1
    
    @patch('builtins.print')
    def test_calculate_rainfall_interval_probability_string_input(self, mock_print):
        """Test with string date input."""
        trace = self.create_mock_trace()
        
        result = calculate_rainfall_interval_probability(trace, "01/15", 0.1, 0.5)
        
        assert result is not None
        assert result['day_of_year'] == 15
        assert result['day_name'] == "January 15"
    
    @patch('builtins.print')
    def test_calculate_rainfall_interval_probability_tuple_input(self, mock_print):
        """Test with tuple date input."""
        trace = self.create_mock_trace()
        
        result = calculate_rainfall_interval_probability(trace, (1, 15), 0.1, 0.5)
        
        assert result is not None
        assert result['day_of_year'] == 15
        assert result['day_name'] == "January 15"
    
    @patch('builtins.print')
    def test_calculate_rainfall_interval_probability_custom_samples(self, mock_print):
        """Test with custom number of samples."""
        trace = self.create_mock_trace()
        
        result = calculate_rainfall_interval_probability(trace, 15, n_samples=500)
        
        assert result is not None
        assert len(result['samples']) <= 500


class TestCalculateAnyRainProbability:
    """Test the calculate_any_rain_probability function."""
    
    def create_mock_trace(self):
        """Create a mock trace for testing."""
        # Create mock posterior data
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 100)),
            'b_rain': np.random.normal(0, 1, (2, 100)),
            'c_rain': np.random.normal(0, 1, (2, 100)),
            'a_amount': np.random.normal(0, 1, (2, 100)),
            'b_amount': np.random.normal(0, 1, (2, 100)),
            'c_amount': np.random.normal(1, 1, (2, 100)),
            'alpha_amount': np.random.gamma(2, 1, (2, 100))
        }
        
        # Create mock trace
        trace = Mock()
        trace.posterior = Mock()
        
        # Set up the posterior data properly
        for param, values in posterior_data.items():
            param_mock = Mock()
            param_mock.values = values
            param_mock.shape = values.shape
            setattr(trace.posterior, param, param_mock)
        
        # Set up __getitem__ to return the parameter mocks
        def getitem_side_effect(key):
            if hasattr(trace.posterior, key):
                return getattr(trace.posterior, key)
            return Mock(values=np.random.normal(0, 1, (2, 100)))
        
        trace.posterior.__getitem__ = getitem_side_effect
        
        return trace
    
    @patch('builtins.print')
    def test_calculate_any_rain_probability_int_input(self, mock_print):
        """Test with integer day input."""
        trace = self.create_mock_trace()
        
        result = calculate_any_rain_probability(trace, 15)
        
        assert result is not None
        assert 'day_of_year' in result
        assert 'mean_probability' in result
        assert 'std_probability' in result
        assert 'ci_95' in result
        assert 'samples' in result
        assert result['day_of_year'] == 15
        assert 0 <= result['mean_probability'] <= 1
        assert result['std_probability'] >= 0
    
    @patch('builtins.print')
    def test_calculate_any_rain_probability_string_input(self, mock_print):
        """Test with string date input."""
        trace = self.create_mock_trace()
        
        result = calculate_any_rain_probability(trace, "01/15")
        
        assert result is not None
        assert result['day_of_year'] == 15
    
    @patch('builtins.print')
    def test_calculate_any_rain_probability_tuple_input(self, mock_print):
        """Test with tuple date input."""
        trace = self.create_mock_trace()
        
        result = calculate_any_rain_probability(trace, (1, 15))
        
        assert result is not None
        assert result['day_of_year'] == 15
    
    @patch('builtins.print')
    def test_calculate_any_rain_probability_custom_samples(self, mock_print):
        """Test with custom number of samples."""
        trace = self.create_mock_trace()
        
        result = calculate_any_rain_probability(trace, 15, n_samples=500)
        
        assert result is not None
        assert len(result['samples']) <= 500
    
    @patch('builtins.print')
    def test_calculate_any_rain_probability_confidence_intervals(self, mock_print):
        """Test that confidence intervals are properly calculated."""
        trace = self.create_mock_trace()
        
        result = calculate_any_rain_probability(trace, 15)
        
        assert 'ci_95' in result
        assert 'ci_90' in result
        assert 'ci_50' in result
        
        # Check that confidence intervals are ordered correctly
        ci_95 = result['ci_95']
        ci_90 = result['ci_90']
        ci_50 = result['ci_50']
        
        assert ci_95[0] <= ci_90[0] <= ci_50[0] <= ci_50[1] <= ci_90[1] <= ci_95[1]
        assert all(0 <= x <= 1 for x in ci_95)
        assert all(0 <= x <= 1 for x in ci_90)
        assert all(0 <= x <= 1 for x in ci_50)


class TestIntegration:
    """Integration tests using real data and model."""
    
    def test_integration_with_real_data(self):
        """Test analysis functions with real data from the model."""
        from bayesian_rainfall.model import load_data, create_rainfall_model, sample_model
        
        # Load real data
        data = load_data("data/noaa_historical_weather_eugene_or_2019-2024.csv")
        
        # Create and sample model
        model = create_rainfall_model(data)
        trace = sample_model(model, draws=100, tune=100)
        
        # Test analyze_single_day with real trace
        with patch('matplotlib.pyplot.show'):
            result = analyze_single_day(trace, data, 15, show_plots=False)
            assert result is not None
            assert result['day_of_year'] == 15
        
        # Test calculate_any_rain_probability with real trace
        with patch('builtins.print'):
            result = calculate_any_rain_probability(trace, 15, n_samples=50)
            assert result is not None
            assert 0 <= result['mean_probability'] <= 1
        
        # Test calculate_rainfall_interval_probability with real trace
        with patch('builtins.print'):
            result = calculate_rainfall_interval_probability(trace, 15, n_samples=50)
            assert result is not None
            assert 0 <= result['probability'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
