from bayesian_rainfall.model import create_model, sample_model


def main():
    """Run the simple Bayesian rainfall analysis."""
    print("Simple Bayesian Rainfall Model - Eugene, OR")
    print("=" * 40)
    
    try:
        # Create model
        model, data = create_model()
        print(f"Data loaded: {len(data)} records")
        
        # Sample from model
        print("Sampling from model...")
        trace = sample_model(model)
        
        print("Analysis completed!")
        print(f"Model parameters: {list(trace.posterior.data_vars.keys())}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
