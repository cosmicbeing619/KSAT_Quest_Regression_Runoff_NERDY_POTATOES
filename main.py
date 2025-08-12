import subprocess

def main():
    print("Step 1: Cleaning and combining raw Excel sheets...")
    subprocess.run(["python", "src/data_cleaning.py"], check=True)

    print("Step 2: Training Random Forest model...")
    subprocess.run(["python", "src/train_rf_model.py"], check=True)

    print("Step 3: Evaluating model on subsets...")
    subprocess.run(["python", "src/evaluate_rf_subsets.py"], check=True)

if __name__ == "__main__":
    main()
