# main.py

import subprocess

def main():
    print("Step 1: Cleaning and combining raw Excel sheets...")
    subprocess.run(["python", "data_cleaning.py"], check=True)

    print("Step 2: Performing feature selection and preprocessing...")
    subprocess.run(["python", "feature_selection.py"], check=True)

    print("Step 3: Running subset training experiments with LightGBM...")
    subprocess.run(["python", "model_training.py"], check=True)

    print("All steps completed successfully!")

if __name__ == "__main__":
    main()
