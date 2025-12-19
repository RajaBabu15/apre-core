
import os
import requests
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def download_esc50_data(url, out_dir):
    """
    Downloads and extracts ESC-50 dataset.
    Returns (audio_dir, meta_path)
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    zip_name = "esc50.zip"
    zip_path = os.path.join(out_dir, zip_name)
    
    # Assuming standard structure of ESC-50-master zip
    repo_root = os.path.join(out_dir, "ESC-50-master")
    audio_dir = os.path.join(repo_root, "audio")
    meta_path = os.path.join(repo_root, "meta", "esc50.csv")
    
    if os.path.exists(audio_dir) and os.path.exists(meta_path):
        print("Data already downloaded and extracted.")
        return audio_dir, meta_path

    if not os.path.exists(zip_path):
        print(f"Downloading ESC-50 from {url}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc=zip_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    size = f.write(chunk)
                    bar.update(size)
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(f"Failed to download data: {e}")

    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
        
    return audio_dir, meta_path

def load_metadata(path):
    print(f"Loading metadata from {path}...")
    return pd.read_csv(path)

def select_classes(meta, candidate_classes, min_selected=5):
    """
    Selects valid classes from candidates that exist in meta.
    """
    print(f"Selecting classes from: {candidate_classes}")
    available = meta['category'].unique()
    valid = [c for c in candidate_classes if c in available]
    
    if len(valid) < min_selected:
        print(f"Warning: Found {len(valid)} classes, but expected at least {min_selected}. Using available.")
    
    print(f"Selected classes: {valid}")
    # Filter meta
    meta_sub = meta[meta['category'].isin(valid)].copy()
    return valid, meta_sub

def prepare_labels(labels):
    """
    Returns (y_encoded, label_encoder, class_names)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    class_names = le.classes_
    return y_encoded, le, class_names
