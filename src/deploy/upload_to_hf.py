# Import required packages
import os

from huggingface_hub import HfApi

# Constants
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
REPO_ID = "akinoshi/mnist-framework-backoff"

# Initialize Hugging Face API
api = HfApi()

# Define the models to upload
uploads = [
    (f"{ROOT_DIR}/src/models/sklearn_rf.joblib", "sklearn_rf.joblib"),
    (f"{ROOT_DIR}/src/models/sklearn_ensemble.joblib", "sklearn_ensemble.joblib"),
    (f"{ROOT_DIR}/src/models/pytorch_cnn.pth", "pytorch_cnn.pth"),
    (f"{ROOT_DIR}/src/models/tensorflow_cnn.keras", "tensorflow_cnn.keras"),
]

for local_path, repo_path in uploads:
    try:
        print(f"Uploading {repo_path} to {REPO_ID}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
            token=os.getenv("HF_TOKEN")  # Ensure you have set your Hugging Face token as an environment variable
        )
        print(f"Uploaded {local_path} to {REPO_ID}/{repo_path}.")
    except Exception as e:
        print(f"Failed to upload {repo_path}: {e}")

print("Model upload completed successfully.")