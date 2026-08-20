"""
Downloads the Online Retail II dataset from the UCI Machine Learning Repository.

Dataset: Online Retail II (ID 502)
Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Citation: Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.
DOI: 10.24432/C5CG6D
License: CC BY 4.0
"""

import os
import zipfile
import requests

RAW_DATA_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join(RAW_DATA_DIR, "online_retail_ii.zip")
EXCEL_PATH = os.path.join(RAW_DATA_DIR, "online_retail_II.xlsx")

UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip"


def download_file(url, destination):
    print(f"Downloading from {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(f"\r  {percent:5.1f}% downloaded", end="")
    print()
    print(f"Saved to {destination}")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction done")


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if os.path.exists(EXCEL_PATH):
        print(f"Raw dataset already present at {EXCEL_PATH}, skipping download.")
        return

    if not os.path.exists(ZIP_PATH):
        try:
            download_file(UCI_ARCHIVE_URL, ZIP_PATH)
        except requests.exceptions.RequestException as error:
            print(f"Download from the official UCI archive failed: {error}")
            print("Could not retrieve the dataset automatically.")
            raise
    else:
        print(f"Zip file already exists at {ZIP_PATH}, skipping download.")

    extract_zip(ZIP_PATH, RAW_DATA_DIR)

    if not os.path.exists(EXCEL_PATH):
        # the archive sometimes contains the file under a slightly different name,
        # so look for any xlsx file in the raw folder and use that instead
        for name in os.listdir(RAW_DATA_DIR):
            if name.lower().endswith(".xlsx"):
                os.rename(os.path.join(RAW_DATA_DIR, name), EXCEL_PATH)
                break

    if os.path.exists(EXCEL_PATH):
        print(f"Dataset ready at {EXCEL_PATH}")
    else:
        raise FileNotFoundError("Could not find the extracted Excel file after unzipping.")


if __name__ == "__main__":
    main()
