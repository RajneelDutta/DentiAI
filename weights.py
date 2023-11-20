import gdown
import os

def weights():
    # Define the Google Drive file ID of the file you want to download
    file_id = "1ND8uss-V0p0EpMBEIItH-jhJd_pfosw1"

    file_path = "best.pt"

    # Check if the file already exists
    if not os.path.exists(file_path):
        # Define the URL for downloading the file from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"

        # Download the file
        gdown.download(url, file_path, quiet=False)
        print(f"File '{file_path}' downloaded successfully.")
    else:
        print(f"File '{file_path}' already exists. Skipping download.")
