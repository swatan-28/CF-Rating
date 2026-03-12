import os
import time

print("Starting automatic ML pipeline...\n")

print("Step 1: Fetching new dataset from Codeforces API...")
os.system("python Model/data_fetch_all_users.py")

time.sleep(2)

print("\nStep 2: Creating subset dataset...")
os.system("python Model/data_subset.py")

time.sleep(2)

print("\nStep 3: Training ML model...")
os.system("python Model/model.py")

print("\nPipeline finished successfully!")
print("New model saved as: regression_model.pkl")