import os
import subprocess
import sys

def run_step(script_name):
    print(f"=========================================")
    print(f"Running {script_name}...")
    print(f"=========================================")
    try:
        # Use simple python command assuming environment is set
        subprocess.check_call([sys.executable, script_name])
        print(f"Successfully ran {script_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)

def main():
    print("Starting CKD Continuous Learning Pipeline...\n")
    
    # Define steps
    steps = [
        "src/download_data.py",      # 1. Get baseline data
        "data/synthetic_generator.py", # 2. Generate new patient data
        "src/preprocessing.py",      # 3. Clean and Prepare
        "src/train.py",              # 4. Train and Evaluate
        "src/explainability.py"      # 5. Verify Explainability
    ]
    
    for step in steps:
        if os.path.exists(step):
            run_step(step)
        else:
            print(f"Warning: {step} not found. Skipping.")

    print("Pipeline Complete! Model is ready for serving.")

if __name__ == "__main__":
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Assuming pipeline.py is in root or src, let's adjust cwd if needed
    # But usually user runs from root.
    main()
