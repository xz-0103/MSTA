import os
import glob
import time
import hashlib
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pywt

_BAND_DEFS = [
    (0.5,  4.0),
    (4.0,  8.0),
    (8.0,  13.0),
    (13.0, 30.0),
    (30.0, 50.0),
]
_WAVENAME  = "morl"
_TOTALSCAL = 51

def _build_scales_and_freqs(fs: int):
    scales = (2 * 1 * _TOTALSCAL) / np.arange(_TOTALSCAL, 1, -1)
    freqs  = pywt.scale2frequency(_WAVENAME, scales) * fs
    return scales, freqs

def _compute_cwt_one(sig: np.ndarray, scales, fs: int, n_freq: int) -> np.ndarray:
    T = len(sig)
    cwt_mat, _ = pywt.cwt(sig, scales, _WAVENAME, 1.0 / fs)
    cwt_mat    = np.abs(cwt_mat).astype(np.float32)
    freqs = pywt.scale2frequency(_WAVENAME, scales) * fs
    result = np.zeros((5, n_freq, T), dtype=np.float32)
    for k, (lo, hi) in enumerate(_BAND_DEFS):
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        if len(idx) == 0:
            continue
        patch = cwt_mat[idx, :]
        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        patch_t = F.adaptive_avg_pool2d(patch_t, (n_freq, T))
        result[k] = patch_t.squeeze(0).squeeze(0).numpy()
    return result

def precompute_cwt_all(signals: np.ndarray, fs: int, n_freq: int) -> np.ndarray:
    N, T      = signals.shape
    scales, _ = _build_scales_and_freqs(fs)
    out       = np.zeros((N, 5, n_freq, T), dtype=np.float32)
    t0 = time.perf_counter()
    for i, sig in enumerate(signals):
        out[i] = _compute_cwt_one(sig, scales, fs, n_freq)
    return out

def _cache_path(root_dir: str, channel_id, sample_size: int,
                threshold: float, fs: int, n_freq: int) -> str:
    key = f"ch{channel_id}_ss{sample_size}_th{threshold}_fs{fs}_nf{n_freq}"
    search_path = os.path.join(root_dir, "**", f"*_{channel_id}.mat")
    files       = sorted(glob.glob(search_path, recursive=True))
    file_hash   = hashlib.md5(str(files).encode()).hexdigest()[:8]
    fname       = f"cwt_{key}_{file_hash}.npz"
    cache_dir   = os.path.join(root_dir, ".cwt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, fname)

class SingleChannelDataset(Dataset):
    def __init__(
        self,
        root_dir:      str,
        channel_id,
        sample_size:   int,
        threshold:     float = 200.0,
        clip_outliers: bool  = True,
        fs:            int   = 200,
        n_freq:        int   = 10,
        use_cache:     bool  = True,
    ):
        self.sample_size   = sample_size
        self.threshold     = threshold
        self.channel_id    = str(channel_id)
        self.clip_outliers = clip_outliers
        self.fs            = fs
        self.n_freq        = n_freq

        search_path    = os.path.join(root_dir, "**", f"*_{self.channel_id}.mat")
        self.file_list = sorted(glob.glob(search_path, recursive=True))
        if not self.file_list:
            raise FileNotFoundError()

        self.mean, self.std = self._calculate_global_stats()
        raw_signals, labels = self._load_and_slice(self.mean, self.std)

        if use_cache:
            cp = _cache_path(root_dir, channel_id, sample_size, threshold, fs, n_freq)
            loaded_from_cache = False
            if os.path.exists(cp):
                try:
                    loaded = np.load(cp)
                    cwt_data = loaded["cwt"]
                    loaded_from_cache = True
                except Exception:
                    os.remove(cp)
            if not loaded_from_cache:
                cwt_data = precompute_cwt_all(raw_signals, fs=fs, n_freq=n_freq)
                np.savez_compressed(cp, cwt=cwt_data)
        else:
            cwt_data = precompute_cwt_all(raw_signals, fs=fs, n_freq=n_freq)

        self.eeg_data = torch.from_numpy(raw_signals[:, np.newaxis, :])
        self.band_data = [
            torch.from_numpy(cwt_data[:, k, np.newaxis, :, :])
            for k in range(5)
        ]
        self.labels = torch.from_numpy(labels)

    def _calculate_global_stats(self):
        valid_points = []
        for fpath in self.file_list:
            eeg = sio.loadmat(fpath)["eeg"][0, :]
            valid_points.append(eeg[np.abs(eeg) <= self.threshold])
        valid_points = np.concatenate(valid_points) if valid_points else np.array([0.0])
        if len(valid_points) == 0:
            return 0.0, 1.0
        return float(np.mean(valid_points)), float(np.std(valid_points))

    def _load_and_slice(self, mean: float, std: float):
        sig_list, lbl_list = [], []
        for fpath in self.file_list:
            mat       = sio.loadmat(fpath)
            eeg_seq   = mat["eeg"][0, :]
            label_seq = mat["eeg"][1, :]
            n_windows = len(eeg_seq) // self.sample_size
            for i in range(n_windows):
                s = i * self.sample_size
                e = s + self.sample_size
                window = eeg_seq[s:e].copy().astype(np.float32)
                if self.clip_outliers:
                    window = np.clip(window, -self.threshold, self.threshold)
                window = (window - mean) / (std + 1e-8)
                sig_list.append(window)
                lbl_list.append(int(label_seq[s]))
        signals = np.stack(sig_list, axis=0)
        labels  = np.array(lbl_list, dtype=np.int64)
        return signals, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg   = self.eeg_data[idx]
        bands = [b[idx] for b in self.band_data]
        label = self.labels[idx]
        return eeg, bands, label

def collate_fn(batch):
    eeg_list   = [item[0] for item in batch]
    bands_list = [item[1] for item in batch]
    lbl_list   = [item[2] for item in batch]
    eeg    = torch.stack(eeg_list,  dim=0)
    label  = torch.stack(lbl_list,  dim=0)
    bands  = [
        torch.stack([bands_list[b][k] for b in range(len(batch))], dim=0)
        for k in range(5)
    ]
    return eeg, bands, label
