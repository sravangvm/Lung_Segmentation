import os
import subprocess

def download_dataset(dataset_name_or_url='tawsifurrahman/covid19-radiography-database', kaggle_json_path="kaggle.json"):
    # Step 1: Install the Kaggle package (if not already installed)
    try:
        import kaggle
    except ImportError:
        print("Kaggle package not found. Installing...")
        subprocess.run(["pip", "install", "kaggle"])

    # Step 2: Copy the kaggle.json file to the .kaggle directory and set permissions
    kaggle_dir = os.path.expanduser("~/.kaggle")

    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)

    subprocess.run(["cp", kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json")])
    subprocess.run(["chmod", "600", os.path.join(kaggle_dir, "kaggle.json")])

    # Step 3: Download the dataset (replace with your desired dataset name or URL)
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name_or_url])

    # Step 4: Unzip the downloaded dataset in the current directory
    subprocess.run(["unzip", f"{dataset_name_or_url.split('/')[-1]}.zip"])

    print("Dataset downloaded and unzipped successfully!")