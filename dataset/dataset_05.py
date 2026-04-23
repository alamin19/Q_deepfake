from huggingface_hub import hf_hub_download
import shutil
import tarfile

def download_and_extract(filename):
    # Download
    file_path = hf_hub_download(
        repo_id="jungjee/asvspoof5",
        repo_type="dataset",
        revision="5d4b1565bc0e3e79343af0b5eacc0ea395405d59",
        filename=filename
    )
    print(f"Downloaded (cache): {file_path}")

    # Copy to /content
    dst = f"/content/{filename}"
    shutil.copy(file_path, dst)
    print(f"Copied to: {dst}")

    # Extract
    extract_dir = f"/content/{filename.replace('.tar','')}"
    with tarfile.open(dst) as tar:
        tar.extractall(extract_dir)
    print(f"Extracted to: {extract_dir}\n")


# -------------------------
# Download both
# -------------------------
download_and_extract("ASVspoof5_protocols.tar")
download_and_extract("flac_T_ab.tar")
