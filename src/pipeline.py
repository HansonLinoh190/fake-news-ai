import os
import argparse
from train import train_model

def run_pipeline(data_path=None, models_dir=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "merged.csv")
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
        
    print("\nStarting Automated ML Pipeline")
    print(f"Dataset path: {data_path}")
    print(f"Models directory: {models_dir}")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found")
        return False
        
    try:
        train_model(data_path, models_dir)
        print("Pipeline Succees!")
        return True
    except Exception as e:
        print(f"Error during pipeline run: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated MLOps Pipeline")
    parser.add_argument("--data", type=str, help="Path to data CSV")
    parser.add_argument("--model-dir", type=str, help="Directory to save model pkl")
    args = parser.parse_args()
    
    run_pipeline(data_path=args.data, models_dir=args.model_dir)
