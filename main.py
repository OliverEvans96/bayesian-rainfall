from bayesian_rainfall.model import load_data, create_model, sample_model
import bayesian_rainfall.analysis as br_analysis
import numpy as np


def main():
    """Run the Bayesian rainfall analysis with year-to-year variation."""
    print("Bayesian Rainfall Model with Year-to-Year Variation - Eugene, OR")
    print("=" * 70)
    
    try:
        # Load data
        print("Loading data...")
        data = load_data("data/noaa_historical_weather_eugene_or_2019-2024.csv")
        print(f"Data loaded: {len(data)} records")
        print(f"Years: {sorted(data['year'].unique())}")
        
        # Create model
        print("Creating hierarchical model...")
        model = create_model(data, n_harmonics=5, include_trend=False)
        
        # Sample from model
        print("Sampling from model...")
        trace = sample_model(model)
        
        print("Analysis completed!")
        print(f"Model parameters: {list(trace.posterior.data_vars.keys())}")
        
        # Demonstrate year-specific analysis
        print("\n" + "="*50)
        print("YEAR-SPECIFIC ANALYSIS EXAMPLE")
        print("="*50)
        
        # Compare a specific day across years
        print("Comparing January 15th across different years:")
        br_analysis.compare_years_for_day(trace, data, "01/15")
        
        # Show year effects
        print("\nPlotting year effects...")
        br_analysis.plot_year_effects(trace, data)
        
        # Demonstrate December rainfall analysis
        print("\n" + "="*50)
        print("DECEMBER RAINFALL ANALYSIS")
        print("="*50)
        
        december_totals = br_analysis.sample_december_total_rainfall(trace, data, n_samples=1000)
        print(f"December total rainfall - Mean: {np.mean(december_totals):.2f} mm")
        print(f"December total rainfall - 95% CI: {np.percentile(december_totals, [2.5, 97.5])}")
        
        # Show comprehensive analysis
        print("\nRunning comprehensive December analysis...")
        december_results = br_analysis.analyze_december_rainfall(trace, data, n_samples=1000, show_plots=True)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
