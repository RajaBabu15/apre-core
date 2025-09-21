import os
import numpy as np
import librosa
import pywt
from tqdm import tqdm

def extract_advanced_features(path, sr=22050, n_mfcc=20, target_seconds=5):

    try:
        y, sr = librosa.load(path, sr=sr)
        target_len = sr * target_seconds
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        features = {}

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features['mfcc_mean'] = mfcc.mean(axis=1)
        features['mfcc_std'] = mfcc.std(axis=1)
        features['mfcc_delta_mean'] = librosa.feature.delta(mfcc).mean(axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = spectral_centroid.mean()
        features['spectral_centroid_std'] = spectral_centroid.std()

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = spectral_bandwidth.mean()
        features['spectral_bandwidth_std'] = spectral_bandwidth.std()

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = spectral_rolloff.mean()
        features['spectral_rolloff_std'] = spectral_rolloff.std()

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = spectral_contrast.mean(axis=1)
        features['spectral_contrast_std'] = spectral_contrast.std(axis=1)

        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = zcr.mean()
        features['zcr_std'] = zcr.std()

        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = rms.mean()
        features['rms_std'] = rms.std()

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = chroma_stft.mean(axis=1)
        features['chroma_stft_std'] = chroma_stft.std(axis=1)

        try:
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
            features['chroma_cqt_mean'] = chroma_cqt.mean(axis=1)
            features['chroma_cqt_std'] = chroma_cqt.std(axis=1)
        except Exception:
            features['chroma_cqt_mean'] = np.zeros(chroma_stft.shape[0])
            features['chroma_cqt_std'] = np.zeros(chroma_stft.shape[0])

        try:
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
            features['chroma_cens_mean'] = chroma_cens.mean(axis=1)
            features['chroma_cens_std'] = chroma_cens.std(axis=1)
        except Exception:
            features['chroma_cens_mean'] = np.zeros(chroma_stft.shape[0])
            features['chroma_cens_std'] = np.zeros(chroma_stft.shape[0])

        try:
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz_mean'] = tonnetz.mean(axis=1)
            features['tonnetz_std'] = tonnetz.std(axis=1)
        except Exception:
            features['tonnetz_mean'] = np.zeros(6)
            features['tonnetz_std'] = np.zeros(6)

        melspec = librosa.feature.melspectrogram(y=y, sr=sr)
        features['melspec_mean'] = melspec.mean(axis=1)
        features['melspec_std'] = melspec.std(axis=1)

        try:
            stft = librosa.stft(y)
            tempo_features = np.abs(stft).mean(axis=1)
            features['tempo'] = float(tempo_features.mean())
        except Exception:
            features['tempo'] = 0.0

        try:
            max_level = pywt.dwt_max_level(data_len=len(y), filter_len=pywt.Wavelet('db4').dec_len)
            use_level = min(5, max_level) if max_level >= 1 else 1
            coeffs = pywt.wavedec(y, 'db4', level=use_level)
            for i, coeff in enumerate(coeffs):
                features[f'wavelet_mean_{i}'] = np.array([np.mean(coeff)])
                features[f'wavelet_std_{i}'] = np.array([np.std(coeff)])
        except Exception:
            for i in range(6):
                features[f'wavelet_mean_{i}'] = np.array([0.0])
                features[f'wavelet_std_{i}'] = np.array([0.0])

        flattened_parts = []
        for key in sorted(features.keys()):
            val = features[key]
            if np.isscalar(val):
                flattened_parts.append(np.array([val], dtype=float))
            else:
                flattened_parts.append(np.array(val, dtype=float).flatten())

        if not flattened_parts:
            raise RuntimeError(f"No features produced for {path}")

        all_features = np.concatenate(flattened_parts)
        return all_features

    except Exception as e:
        print(f"Error processing {path}: {e}")

        return np.zeros(282)

def _process_row(row, audio_dir):

    fname = row['filename']
    path = os.path.join(audio_dir, fname)
    if not os.path.exists(path):
        return None
    try:
        feats = extract_advanced_features(path)
        return (feats, row['category'], path)
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return None

def extract_features_from_metadata(meta_sub, audio_dir):

    print("Extracting advanced features (sequential processing to avoid segfaults)...")

    features, labels, filepaths = [], [], []

    for _, row in tqdm(meta_sub.iterrows(), total=len(meta_sub), desc="Processing audio files"):
        r = _process_row(row, audio_dir)
        if r is None:
            continue
        feats, lbl, p = r
        features.append(feats)
        labels.append(lbl)
        filepaths.append(p)

    if len(features) == 0:
        raise RuntimeError("No features extracted, aborting.")

    max_len = max(f.shape[0] for f in features)
    X = np.array([np.pad(f, (0, max_len - f.shape[0])) for f in features])

    return X, labels, filepaths
