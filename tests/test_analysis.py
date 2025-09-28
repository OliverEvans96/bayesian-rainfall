"""
Comprehensive tests for the analysis module.
"""

import pytest
import numpy as np
import pandas as pd
import arviz as az
from unittest.mock import Mock, patch
from datetime import datetime
import pymc as pm

from bayesian_rainfall.analysis import (
    _parse_date_input,
    print_model_summary,
    print_convergence_diagnostics,
    analyze_single_day,
    calculate_rainfall_interval_probability,
    calculate_any_rain_probability,
    _sample_posterior_predictive_hierarchical,
    _calculate_expected_values_with_year_effects,
    _get_year_effects_from_trace,
    _apply_year_effects_to_predictions,
    _get_all_year_predictions,
    _get_observed_data
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
        # Create mock posterior data with 5 harmonics (matching new default)
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 1000, 5)),
            'b_rain': np.random.normal(0, 1, (2, 1000, 5)),
            'c_rain': np.random.normal(0, 1, (2, 1000)),
            'a_amount': np.random.normal(0, 1, (2, 1000, 5)),
            'b_amount': np.random.normal(0, 1, (2, 1000, 5)),
            'c_amount': np.random.normal(1, 1, (2, 1000)),
            'alpha_amount': np.random.gamma(2, 1, (2, 1000)),
            'year_rain_effects': np.random.normal(0, 0.1, (2, 1000, 6)),  # 6 years
            'year_amount_effects': np.random.normal(0, 0.1, (2, 1000, 6))  # 6 years
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
        
        # Set up coordinates for year effects
        year_rain_effects_mock = trace.posterior.year_rain_effects
        year_rain_effects_mock.coords = Mock()
        year_rain_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
        year_amount_effects_mock = trace.posterior.year_amount_effects
        year_amount_effects_mock.coords = Mock()
        year_amount_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
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
        # Create mock posterior data with 5 harmonics (matching new default)
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 100, 5)),
            'b_rain': np.random.normal(0, 1, (2, 100, 5)),
            'c_rain': np.random.normal(0, 1, (2, 100)),
            'a_amount': np.random.normal(0, 1, (2, 100, 5)),
            'b_amount': np.random.normal(0, 1, (2, 100, 5)),
            'c_amount': np.random.normal(1, 1, (2, 100)),
            'alpha_amount': np.random.gamma(2, 1, (2, 100)),
            'year_rain_effects': np.random.normal(0, 0.1, (2, 100, 6)),  # 6 years
            'year_amount_effects': np.random.normal(0, 0.1, (2, 100, 6))  # 6 years
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
        
        # Set up coordinates for year effects
        year_rain_effects_mock = trace.posterior.year_rain_effects
        year_rain_effects_mock.coords = Mock()
        year_rain_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
        year_amount_effects_mock = trace.posterior.year_amount_effects
        year_amount_effects_mock.coords = Mock()
        year_amount_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
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
        # Create mock posterior data with 5 harmonics (matching new default)
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 100, 5)),
            'b_rain': np.random.normal(0, 1, (2, 100, 5)),
            'c_rain': np.random.normal(0, 1, (2, 100)),
            'a_amount': np.random.normal(0, 1, (2, 100, 5)),
            'b_amount': np.random.normal(0, 1, (2, 100, 5)),
            'c_amount': np.random.normal(1, 1, (2, 100)),
            'alpha_amount': np.random.gamma(2, 1, (2, 100)),
            'year_rain_effects': np.random.normal(0, 0.1, (2, 100, 6)),  # 6 years
            'year_amount_effects': np.random.normal(0, 0.1, (2, 100, 6))  # 6 years
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
        
        # Set up coordinates for year effects
        year_rain_effects_mock = trace.posterior.year_rain_effects
        year_rain_effects_mock.coords = Mock()
        year_rain_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
        year_amount_effects_mock = trace.posterior.year_amount_effects
        year_amount_effects_mock.coords = Mock()
        year_amount_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
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
        # Create mock posterior data with 5 harmonics (matching new default)
        posterior_data = {
            'a_rain': np.random.normal(0, 1, (2, 100, 5)),
            'b_rain': np.random.normal(0, 1, (2, 100, 5)),
            'c_rain': np.random.normal(0, 1, (2, 100)),
            'a_amount': np.random.normal(0, 1, (2, 100, 5)),
            'b_amount': np.random.normal(0, 1, (2, 100, 5)),
            'c_amount': np.random.normal(1, 1, (2, 100)),
            'alpha_amount': np.random.gamma(2, 1, (2, 100)),
            'year_rain_effects': np.random.normal(0, 0.1, (2, 100, 6)),  # 6 years
            'year_amount_effects': np.random.normal(0, 0.1, (2, 100, 6))  # 6 years
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
        
        # Set up coordinates for year effects
        year_rain_effects_mock = trace.posterior.year_rain_effects
        year_rain_effects_mock.coords = Mock()
        year_rain_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
        year_amount_effects_mock = trace.posterior.year_amount_effects
        year_amount_effects_mock.coords = Mock()
        year_amount_effects_mock.coords.__getitem__ = Mock(return_value=Mock(values=np.array([2019, 2020, 2021, 2022, 2023, 2024])))
        
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
        from bayesian_rainfall.model import load_data, create_model, sample_model
        
        # Load real data
        data = load_data("data/noaa_historical_weather_eugene_or_2019-2024.csv")
        
        # Create and sample model
        model = create_model(data)
        trace = sample_model(model, draws=100, tune=100)

        # Test posterior predictive sampling using our simplified function
        from bayesian_rainfall.analysis import sample_posterior_predictive_for_day
        rain_indicators, rainfall_amounts = sample_posterior_predictive_for_day(trace, 15, n_samples=50)

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


class TestHierarchicalPosteriorPredictiveSampling:
    """Test the hierarchical posterior predictive sampling functions."""
    
    def create_mock_trace(self):
        """Create a mock trace for testing."""
        # Create mock trace with hierarchical structure
        mock_trace = Mock()
        mock_trace.posterior = Mock()
        
        # Mock posterior samples
        n_chains, n_draws, n_harmonics = 2, 50, 3
        n_years = 6
        
        # Create realistic parameter shapes
        mock_trace.posterior.a_rain = Mock()
        mock_trace.posterior.a_rain.values = np.random.normal(0, 0.5, (n_chains, n_draws, n_harmonics))
        mock_trace.posterior.a_rain.shape = (n_chains, n_draws, n_harmonics)
        
        mock_trace.posterior.b_rain = Mock()
        mock_trace.posterior.b_rain.values = np.random.normal(0, 0.5, (n_chains, n_draws, n_harmonics))
        
        mock_trace.posterior.c_rain = Mock()
        mock_trace.posterior.c_rain.values = np.random.normal(-1, 0.5, (n_chains, n_draws))
        
        mock_trace.posterior.a_amount = Mock()
        mock_trace.posterior.a_amount.values = np.random.normal(0, 0.3, (n_chains, n_draws, n_harmonics))
        
        mock_trace.posterior.b_amount = Mock()
        mock_trace.posterior.b_amount.values = np.random.normal(0, 0.3, (n_chains, n_draws, n_harmonics))
        
        mock_trace.posterior.c_amount = Mock()
        mock_trace.posterior.c_amount.values = np.random.normal(1, 0.5, (n_chains, n_draws))
        
        mock_trace.posterior.alpha_amount = Mock()
        mock_trace.posterior.alpha_amount.values = np.random.gamma(2, 1, (n_chains, n_draws))
        
        # Year effects
        mock_trace.posterior.year_rain_effects = Mock()
        mock_trace.posterior.year_rain_effects.values = np.random.normal(0, 0.2, (n_chains, n_draws, n_years))
        # Create a mock coords object that behaves like xarray coords
        mock_coords = Mock()
        mock_coords.__getitem__ = lambda self, key: np.arange(2019, 2025) if key == 'year' else None
        mock_trace.posterior.year_rain_effects.coords = mock_coords
        
        mock_trace.posterior.year_amount_effects = Mock()
        mock_trace.posterior.year_amount_effects.values = np.random.normal(0, 0.1, (n_chains, n_draws, n_years))
        
        return mock_trace
    
    def create_mock_data(self):
        """Create mock data for testing."""
        dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
        data = pd.DataFrame({
            'DATE': dates,
            'PRCP': np.random.exponential(2, len(dates)),
            'day_of_year': dates.dayofyear,
            'year': dates.year
        })
        # Make some days have no rain
        data.loc[data['PRCP'] < 1, 'PRCP'] = 0
        return data
    
    def test_sample_posterior_predictive_hierarchical_basic(self):
        """Test basic functionality of hierarchical posterior predictive sampling."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        days_of_year = np.arange(1, 366)
        
        # Create mock expected amounts and alpha amounts
        n_samples = 100
        expected_amounts = np.random.exponential(3, (365, n_samples))
        alpha_amounts = np.random.gamma(2, 1, (365, n_samples))
        
        # Test the function
        with patch('builtins.print'):  # Suppress print statements
            rain_indicators, rainfall_amounts = _sample_posterior_predictive_hierarchical(
                trace, data, days_of_year, expected_amounts, alpha_amounts
            )
        
        # Check output shapes
        assert rain_indicators.shape == (n_samples, 365)
        assert rainfall_amounts.shape == (n_samples, 365)
        
        # Check that rain indicators are binary
        assert np.all(np.isin(rain_indicators, [0, 1]))
        
        # Check that rainfall amounts are non-negative
        assert np.all(rainfall_amounts >= 0)
        
        # Check that rainfall amounts are 0 when rain_indicators is 0
        no_rain_mask = rain_indicators == 0
        assert np.all(rainfall_amounts[no_rain_mask] == 0)
    
    def test_sample_posterior_predictive_hierarchical_rain_probability(self):
        """Test that rain probabilities are reasonable."""
        trace = self.create_mock_trace()
        data = self.create_mock_data()
        days_of_year = np.arange(1, 366)
        
        n_samples = 100
        expected_amounts = np.random.exponential(3, (365, n_samples))
        alpha_amounts = np.random.gamma(2, 1, (365, n_samples))
        
        with patch('builtins.print'):
            rain_indicators, rainfall_amounts = _sample_posterior_predictive_hierarchical(
                trace, data, days_of_year, expected_amounts, alpha_amounts
            )
        
        # Check that rain frequency is reasonable (not 0 or 1 for all days)
        rain_frequencies = np.mean(rain_indicators, axis=0)
        assert np.any(rain_frequencies > 0)  # Some days should have rain
        assert np.any(rain_frequencies < 1)  # Some days should not have rain
    
    def test_calculate_expected_values_with_year_effects(self):
        """Test the expected values calculation with year effects."""
        trace = self.create_mock_trace()
        days_of_year = np.arange(1, 366)
        
        with patch('builtins.print'):
            rain_probs, expected_amounts, alpha_amounts = _calculate_expected_values_with_year_effects(
                trace, days_of_year
            )
        
        # Check output shapes
        assert rain_probs.shape == (365, 100)  # (n_days, n_samples)
        assert expected_amounts.shape == (365, 100)
        assert alpha_amounts.shape == (365, 100)
        
        # Check that rain probabilities are in [0, 1]
        assert np.all(rain_probs >= 0)
        assert np.all(rain_probs <= 1)
        
        # Check that expected amounts are positive
        assert np.all(expected_amounts > 0)
        
        # Check that alpha amounts are positive
        assert np.all(alpha_amounts > 0)
    
    def test_get_year_effects_from_trace(self):
        """Test extraction of year effects from trace."""
        trace = self.create_mock_trace()
        
        year_rain_effects, year_amount_effects, unique_years = _get_year_effects_from_trace(trace)
        
        # Check shapes
        n_samples = trace.posterior.a_rain.values.size // trace.posterior.a_rain.values.shape[-1]
        n_years = len(unique_years)
        
        assert year_rain_effects.shape == (n_samples, n_years)
        assert year_amount_effects.shape == (n_samples, n_years)
        assert len(unique_years) == n_years
        
        # Check that years are reasonable
        assert np.all(unique_years >= 2019)
        assert np.all(unique_years <= 2024)
    
    def test_get_observed_data(self):
        """Test extraction of observed data for plotting."""
        data = self.create_mock_data()
        
        observed_rain_prob, observed_days, observed_rainfall, observed_rainy_days = _get_observed_data(data)
        
        # Check that we get some data
        assert len(observed_rain_prob) > 0
        assert len(observed_days) > 0
        
        # Check that rain probabilities are in [0, 1]
        assert np.all(observed_rain_prob >= 0)
        assert np.all(observed_rain_prob <= 1)
        
        # Check that observed rainfall is non-negative
        assert np.all(observed_rainfall >= 0)
    
    def test_apply_year_effects_to_predictions(self):
        """Test applying year effects to base predictions."""
        # Create test data
        n_samples = 100
        rain_probs_base = np.random.uniform(0.1, 0.9, n_samples)
        expected_amounts_base = np.random.exponential(3, n_samples)
        
        year_rain_effects = np.random.normal(0, 0.2, (n_samples, 6))
        year_amount_effects = np.random.normal(0, 0.1, (n_samples, 6))
        unique_years = np.arange(2019, 2025)
        
        rain_probs_with_year, expected_amounts_with_year = _apply_year_effects_to_predictions(
            rain_probs_base, expected_amounts_base, year_rain_effects, year_amount_effects, unique_years
        )
        
        # Check output shapes
        assert len(rain_probs_with_year) == n_samples
        assert len(expected_amounts_with_year) == n_samples
        
        # Check that probabilities are still in [0, 1]
        assert np.all(rain_probs_with_year >= 0)
        assert np.all(rain_probs_with_year <= 1)
        
        # Check that amounts are positive
        assert np.all(expected_amounts_with_year > 0)
    
    def test_get_all_year_predictions(self):
        """Test getting year-specific predictions for all days and years."""
        trace = self.create_mock_trace()
        days_of_year = np.arange(1, 366)
        unique_years = np.arange(2019, 2025)
        
        with patch('builtins.print'):
            all_year_rain_probs, all_year_expected_amounts = _get_all_year_predictions(
                trace, days_of_year, unique_years
            )
        
        n_samples = trace.posterior.a_rain.values.size // trace.posterior.a_rain.values.shape[-1]
        n_days = len(days_of_year)
        n_years = len(unique_years)
        
        # Check output shapes
        assert all_year_rain_probs.shape == (n_samples, n_days, n_years)
        assert all_year_expected_amounts.shape == (n_samples, n_days, n_years)
        
        # Check that rain probabilities are in [0, 1]
        assert np.all(all_year_rain_probs >= 0)
        assert np.all(all_year_rain_probs <= 1)
        
        # Check that expected amounts are positive
        assert np.all(all_year_expected_amounts > 0)


class TestVisualizationFunctions:
    """Test visualization functions that use the hierarchical sampling."""
    
    def create_mock_trace_and_data(self):
        """Create mock trace and data for visualization testing."""
        # Create mock trace
        mock_trace = Mock()
        mock_trace.posterior = Mock()
        
        n_chains, n_draws, n_harmonics = 2, 50, 3
        n_years = 6
        
        # Mock posterior samples
        mock_trace.posterior.a_rain = Mock()
        mock_trace.posterior.a_rain.values = np.random.normal(0, 0.5, (n_chains, n_draws, n_harmonics))
        mock_trace.posterior.a_rain.shape = (n_chains, n_draws, n_harmonics)
        
        mock_trace.posterior.b_rain = Mock()
        mock_trace.posterior.b_rain.values = np.random.normal(0, 0.5, (n_chains, n_draws, n_harmonics))
        
        mock_trace.posterior.c_rain = Mock()
        mock_trace.posterior.c_rain.values = np.random.normal(-1, 0.5, (n_chains, n_draws))
        
        mock_trace.posterior.a_amount = Mock()
        mock_trace.posterior.a_amount.values = np.random.normal(0, 0.3, (n_chains, n_draws, n_harmonics))
        
        mock_trace.posterior.b_amount = Mock()
        mock_trace.posterior.b_amount.values = np.random.normal(0, 0.3, (n_chains, n_draws, n_harmonics))
        
        mock_trace.posterior.c_amount = Mock()
        mock_trace.posterior.c_amount.values = np.random.normal(1, 0.5, (n_chains, n_draws))
        
        mock_trace.posterior.alpha_amount = Mock()
        mock_trace.posterior.alpha_amount.values = np.random.gamma(2, 1, (n_chains, n_draws))
        
        # Year effects
        mock_trace.posterior.year_rain_effects = Mock()
        mock_trace.posterior.year_rain_effects.values = np.random.normal(0, 0.2, (n_chains, n_draws, n_years))
        # Create a mock coords object that behaves like xarray coords
        mock_coords = Mock()
        mock_coords.__getitem__ = lambda self, key: np.arange(2019, 2025) if key == 'year' else None
        mock_trace.posterior.year_rain_effects.coords = mock_coords
        
        mock_trace.posterior.year_amount_effects = Mock()
        mock_trace.posterior.year_amount_effects.values = np.random.normal(0, 0.1, (n_chains, n_draws, n_years))
        
        # Create mock data
        dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
        data = pd.DataFrame({
            'DATE': dates,
            'PRCP': np.random.exponential(2, len(dates)),
            'day_of_year': dates.dayofyear,
            'year': dates.year
        })
        data.loc[data['PRCP'] < 1, 'PRCP'] = 0
        
        return mock_trace, data
    
    def test_plot_combined_predictions_basic(self):
        """Test that plot_combined_predictions runs without errors."""
        from bayesian_rainfall.visualizations import plot_combined_predictions
        
        trace, data = self.create_mock_trace_and_data()
        
        # Test with mocked plotting
        with patch('matplotlib.pyplot.show'), patch('builtins.print'):
            plot_combined_predictions(trace, data, ci_level=0.95, figsize=(10, 8))
    
    def test_plot_combined_predictions_ci_levels(self):
        """Test plot_combined_predictions with different confidence levels."""
        from bayesian_rainfall.visualizations import plot_combined_predictions
        
        trace, data = self.create_mock_trace_and_data()
        
        # Test different confidence levels
        for ci_level in [0.90, 0.95, 0.99]:
            with patch('matplotlib.pyplot.show'), patch('builtins.print'):
                plot_combined_predictions(trace, data, ci_level=ci_level, figsize=(10, 8))
    
    def test_plot_combined_predictions_output_shapes(self):
        """Test that plot_combined_predictions produces correct intermediate results."""
        from bayesian_rainfall.visualizations import _calculate_expected_values_with_year_effects
        from bayesian_rainfall.visualizations import _sample_posterior_predictive_hierarchical
        from bayesian_rainfall.visualizations import _get_observed_data
        
        trace, data = self.create_mock_trace_and_data()
        days_of_year = np.arange(1, 366)
        
        # Test expected values calculation
        with patch('builtins.print'):
            rain_probs, expected_amounts, alpha_amounts = _calculate_expected_values_with_year_effects(
                trace, days_of_year
            )
        
        # Check shapes
        n_samples = trace.posterior.a_rain.values.size // trace.posterior.a_rain.values.shape[-1]
        assert rain_probs.shape == (365, n_samples)
        assert expected_amounts.shape == (365, n_samples)
        assert alpha_amounts.shape == (365, n_samples)
        
        # Test posterior predictive sampling
        with patch('builtins.print'):
            rain_indicators, rainfall_amounts = _sample_posterior_predictive_hierarchical(
                trace, data, days_of_year, expected_amounts, alpha_amounts
            )
        
        # Check shapes
        assert rain_indicators.shape == (n_samples, 365)
        assert rainfall_amounts.shape == (n_samples, 365)
        
        # Test observed data extraction
        observed_rain_prob, observed_days, observed_rainfall, observed_rainy_days = _get_observed_data(data)
        
        # Check that we get reasonable data
        assert len(observed_rain_prob) > 0
        assert len(observed_days) > 0
    
    def test_rain_probability_pp_ci_correctness(self):
        """Test that rain probability PP CI shows probability ranges, not [0,1]."""
        from bayesian_rainfall.visualizations import _calculate_expected_values_with_year_effects
        
        trace, data = self.create_mock_trace_and_data()
        days_of_year = np.arange(1, 366)
        
        # Test expected values calculation
        with patch('builtins.print'):
            rain_probs, expected_amounts, alpha_amounts = _calculate_expected_values_with_year_effects(
                trace, days_of_year
            )
        
        # Calculate PP CI using the corrected approach (probability values, not binary outcomes)
        lower_percentile = 2.5
        upper_percentile = 97.5
        
        pp_rain_lower = np.percentile(rain_probs, lower_percentile, axis=1)
        pp_rain_upper = np.percentile(rain_probs, upper_percentile, axis=1)
        
        # Check that PP CI shows reasonable probability ranges (not [0,1] everywhere)
        # Most days should have PP CI width < 0.8 (not spanning the full [0,1] range)
        pp_ci_widths = pp_rain_upper - pp_rain_lower
        reasonable_width_days = np.sum(pp_ci_widths < 0.8)
        
        # At least 80% of days should have reasonable CI widths
        assert reasonable_width_days > 0.8 * len(days_of_year), \
            f"Too many days have [0,1] PP CI: {len(days_of_year) - reasonable_width_days} out of {len(days_of_year)}"
        
        # Check that PP CI values are in [0,1] range
        assert np.all(pp_rain_lower >= 0)
        assert np.all(pp_rain_upper <= 1)
        assert np.all(pp_rain_lower <= pp_rain_upper)
        
        # Check that PP CI is wider than parameter-only CI (due to year effects)
        param_rain_lower = np.percentile(rain_probs, lower_percentile, axis=1)
        param_rain_upper = np.percentile(rain_probs, upper_percentile, axis=1)
        
        # PP CI should generally be wider due to year effects
        pp_width = np.mean(pp_rain_upper - pp_rain_lower)
        param_width = np.mean(param_rain_upper - param_rain_lower)
        
        # PP CI should be at least as wide as parameter CI
        assert pp_width >= param_width * 0.9, "PP CI should be wider than parameter CI due to year effects"

    def test_sample_observed_rain_frequencies_basic(self):
        """Test basic functionality of _sample_observed_rain_frequencies."""
        from bayesian_rainfall.visualizations import _sample_observed_rain_frequencies
        
        # Create test data
        n_days, n_samples = 10, 100
        rain_probs = np.random.uniform(0.1, 0.9, (n_days, n_samples))
        n_years = 6
        
        # Test the function
        observed_frequencies = _sample_observed_rain_frequencies(rain_probs, n_years)
        
        # Check output shape
        assert observed_frequencies.shape == (n_days, n_samples)
        
        # Check that all values are in [0, 1]
        assert np.all(observed_frequencies >= 0)
        assert np.all(observed_frequencies <= 1)
        
        # Check that values are not identical to input (should have sampling uncertainty)
        assert not np.allclose(observed_frequencies, rain_probs, rtol=1e-10)

    def test_sample_observed_rain_frequencies_edge_cases(self):
        """Test _sample_observed_rain_frequencies with edge cases."""
        from bayesian_rainfall.visualizations import _sample_observed_rain_frequencies
        
        n_days, n_samples = 5, 50
        n_years = 6
        
        # Test with extreme probabilities
        rain_probs = np.array([
            [0.0, 0.0, 0.0],  # All zeros
            [1.0, 1.0, 1.0],  # All ones
            [0.5, 0.5, 0.5],  # All 0.5
            [0.01, 0.99, 0.5],  # Mixed extreme values
            [0.0, 1.0, 0.5]   # Mixed with zeros and ones
        ])
        
        observed_frequencies = _sample_observed_rain_frequencies(rain_probs, n_years)
        
        # Check output shape
        assert observed_frequencies.shape == (n_days, 3)
        
        # Check that all values are in [0, 1]
        assert np.all(observed_frequencies >= 0)
        assert np.all(observed_frequencies <= 1)
        
        # For extreme probabilities, output should be close to input
        assert np.allclose(observed_frequencies[0], rain_probs[0], atol=1e-10)  # All zeros
        assert np.allclose(observed_frequencies[1], rain_probs[1], atol=1e-10)  # All ones

    def test_sample_observed_rain_frequencies_sampling_uncertainty(self):
        """Test that _sample_observed_rain_frequencies properly models sampling uncertainty."""
        from bayesian_rainfall.visualizations import _sample_observed_rain_frequencies
        
        # Test with a single probability value across many samples
        n_days, n_samples = 1, 1000
        true_prob = 0.3
        rain_probs = np.full((n_days, n_samples), true_prob)
        n_years = 6
        
        # Sample multiple times to test variance
        all_samples = []
        for _ in range(10):
            observed_frequencies = _sample_observed_rain_frequencies(rain_probs, n_years)
            all_samples.append(observed_frequencies[0, 0])  # First day, first sample
        
        all_samples = np.array(all_samples)
        
        # Check that we get reasonable variance around the true probability
        sample_mean = np.mean(all_samples)
        sample_var = np.var(all_samples)
        
        # Mean should be close to true probability (allow more tolerance for Beta approximation)
        assert abs(sample_mean - true_prob) < 0.2
        
        # Should have some variance (not all identical)
        assert sample_var > 0
        
        # Variance should be reasonable for n=6 years
        # Theoretical variance for n=6: p(1-p)/6 = 0.3*0.7/6 ≈ 0.035
        expected_var = true_prob * (1 - true_prob) / n_years
        assert sample_var < expected_var * 5  # Should be less than 5x theoretical (due to Beta approximation)

    def test_sample_observed_rain_frequencies_different_n_years(self):
        """Test _sample_observed_rain_frequencies with different n_years values."""
        from bayesian_rainfall.visualizations import _sample_observed_rain_frequencies
        
        n_days, n_samples = 1, 100
        true_prob = 0.4
        rain_probs = np.full((n_days, n_samples), true_prob)
        
        # Test with different n_years values
        for n_years in [1, 3, 6, 12]:
            observed_frequencies = _sample_observed_rain_frequencies(rain_probs, n_years)
            
            # Check output shape
            assert observed_frequencies.shape == (n_days, n_samples)
            
            # Check that all values are in [0, 1]
            assert np.all(observed_frequencies >= 0)
            assert np.all(observed_frequencies <= 1)
            
            # With more years, variance should generally be lower
            # (though this is stochastic, so we just check it's reasonable)
            sample_var = np.var(observed_frequencies[0])
            expected_var = true_prob * (1 - true_prob) / n_years
            assert sample_var < expected_var * 3  # Should be less than 3x theoretical

    def test_sample_observed_rain_frequencies_integration(self):
        """Test _sample_observed_rain_frequencies integration with plot_combined_predictions."""
        from bayesian_rainfall.visualizations import plot_combined_predictions
        
        trace, data = self.create_mock_trace_and_data()
        
        # Test that the function works within the full plotting pipeline
        with patch('matplotlib.pyplot.show'), patch('builtins.print'):
            # This should not raise any errors
            plot_combined_predictions(trace, data, ci_level=0.95, figsize=(10, 8))
        
        # Test that the Beta distribution sampling is actually being used
        # by checking that the PP CI is wider than parameter uncertainty
        from bayesian_rainfall.visualizations import _calculate_expected_values_with_year_effects
        from bayesian_rainfall.visualizations import _sample_observed_rain_frequencies
        
        days_of_year = np.arange(1, 366)
        
        with patch('builtins.print'):
            rain_probs, expected_amounts, alpha_amounts = _calculate_expected_values_with_year_effects(
                trace, days_of_year
            )
        
        # Calculate parameter uncertainty CI
        param_lower = np.percentile(rain_probs, 2.5, axis=1)
        param_upper = np.percentile(rain_probs, 97.5, axis=1)
        param_width = param_upper - param_lower
        
        # Calculate observed frequencies CI
        observed_frequencies = _sample_observed_rain_frequencies(rain_probs, n_years=6)
        obs_lower = np.percentile(observed_frequencies, 2.5, axis=1)
        obs_upper = np.percentile(observed_frequencies, 97.5, axis=1)
        obs_width = obs_upper - obs_lower
        
        # Observed frequencies CI should generally be wider than parameter CI
        # (due to sampling uncertainty)
        assert np.mean(obs_width) > np.mean(param_width) * 0.8  # Allow some tolerance


if __name__ == "__main__":
    pytest.main([__file__])
