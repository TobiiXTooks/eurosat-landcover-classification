"""
This module is ONLY for loading the dataset in a numpy way for the from scratch implementation of the neural network
"""

import os
import time
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lib.constants import PATH_TO_DATASET, S2_REFLECTANCE_SCALE

IMG_SIZE = 64
N_BANDS = 13  # Sentinel-2 / EuroSAT MS


def load_and_preprocess_image(
    path_to_image: str, size: int = IMG_SIZE
) -> npt.NDArray[np.float64]:
    """Read a EuroSAT MS .tif, keep all 13 bands, normalize to [0, 1], flatten.

    Output shape: (size * size * N_BANDS,). For the default 64×64×13 that's 53248
    features per sample, ready to feed an MLP. Throwing away the multispectral
    info (e.g. by grayscaling) is the main reason naive models underperform on
    EuroSAT, so we keep every band.
    """
    # np.asarray forces the loose skimage return types into a concrete ndarray
    # so pyright recognises .astype / .shape / etc.
    img = np.asarray(io.imread(path_to_image))  # (H, W, 13), uint16

    # Resize each band to (size, size). skimage handles arbitrary channel counts
    # natively (unlike Pillow). preserve_range keeps reflectance values in their
    # original 0..65535 scale so we can apply S2's normalization below.
    if img.shape[0] != size or img.shape[1] != size:
        img = np.asarray(
            resize(
                img,
                (size, size, img.shape[2]),
                preserve_range=True,
                anti_aliasing=True,
            )
        )

    # Standard EuroSAT scaling: typical surface reflectance / 10000 → [0, 1].
    # Clip the long tail of very-bright pixels (clouds, sun glint) so all values
    # land in [0, 1] regardless of band.
    img = np.clip(img.astype(np.float64) / S2_REFLECTANCE_SCALE, 0.0, 1.0)

    return img.flatten()


class EurosatRawDataset:
    X_train: npt.NDArray[np.float64]  # shape (n_train, IMG_SIZE * IMG_SIZE * N_BANDS)
    X_test: npt.NDArray[np.float64]  # shape (n_test,  IMG_SIZE * IMG_SIZE * N_BANDS)
    y_train: npt.NDArray[np.int64]
    y_test: npt.NDArray[np.int64]
    label_encoder: LabelEncoder

    def __init__(
        self,
        base_dataset_path: Path = PATH_TO_DATASET,
        test_size: float = 0.2,
        random_state: int = 42,
        use_cache: bool = True,
    ) -> None:
        self._base_dataset_path = Path(base_dataset_path)
        self._test_size = test_size
        self._random_state = random_state
        self._use_cache = use_cache

        if use_cache and self._cache_path().exists():
            self._load_from_cache()
        else:
            self._load_dataset()
            if use_cache:
                self._save_to_cache()

    def _cache_path(self) -> Path:
        """Cache key includes the params that affect the output, so changing any of
        them gives a fresh cache file rather than silently returning stale data."""
        cache_dir = self._base_dataset_path.parent / ".cache"
        n_files = (
            sum(1 for _ in self._base_dataset_path.rglob("*.tif"))
            if self._base_dataset_path.exists()
            else 0
        )
        name = (
            f"eurosat_ms_size{IMG_SIZE}"
            f"_bands{N_BANDS}"
            f"_split{self._test_size}"
            f"_seed{self._random_state}"
            f"_n{n_files}.npz"
        )
        return cache_dir / name

    def _load_from_cache(self) -> None:
        path = self._cache_path()
        print(f"[cache] loading preprocessed dataset from {path}")
        t0 = time.perf_counter()
        data = np.load(path, allow_pickle=False)
        self.X_train = data["X_train"]
        self.X_test = data["X_test"]
        self.y_train = data["y_train"]
        self.y_test = data["y_test"]
        self.label_encoder = LabelEncoder()
        # LabelEncoder is fully described by .classes_ — re-attach it and the
        # encoder is ready to .transform / .inverse_transform without re-fitting.
        self.label_encoder.classes_ = data["classes"]
        print(f"[cache] loaded in {time.perf_counter() - t0:.2f}s")

    def _save_to_cache(self) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[cache] saving preprocessed dataset to {path}")
        np.savez_compressed(
            path,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            classes=cast(npt.NDArray, self.label_encoder.classes_),
        )

    def _load_dataset(self):
        """Load every .tif under `base_dataset_path`, encode labels, and split into train/test."""
        print("[load] scanning directory for .tif files…")
        class_names = sorted(os.listdir(self._base_dataset_path))
        image_paths: list[str] = []
        labels: list[str] = []

        for class_name in class_names:
            class_dir = os.path.join(self._base_dataset_path, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(".tif"):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(class_name)

        n = len(image_paths)
        print(f"[load] found {n} images across {len(class_names)} classes")

        print(f"[load] reading + preprocessing {n} images…")
        t0 = time.perf_counter()
        # Print progress every ~5% so the user has feedback on a long run.
        step = max(1, n // 20)
        images: list[npt.NDArray[np.float64]] = []
        for i, path in enumerate(image_paths):
            images.append(load_and_preprocess_image(path))
            if (i + 1) % step == 0 or i + 1 == n:
                pct = 100 * (i + 1) / n
                elapsed = time.perf_counter() - t0
                print(f"  {i + 1}/{n}  ({pct:5.1f}%)  elapsed {elapsed:5.1f}s")
        X: npt.NDArray[np.float64] = np.array(images)
        print(
            f"[load] preprocessing done in {time.perf_counter() - t0:.2f}s, X shape={X.shape}"
        )

        print("[load] encoding labels and splitting train/test…")
        self.label_encoder = LabelEncoder()
        y = cast(npt.NDArray[np.int64], self.label_encoder.fit_transform(labels))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self._test_size, random_state=self._random_state, stratify=y
        )

        self.X_train = cast(npt.NDArray[np.float64], X_train)
        self.X_test = cast(npt.NDArray[np.float64], X_test)
        self.y_train = cast(npt.NDArray[np.int64], y_train)
        self.y_test = cast(npt.NDArray[np.int64], y_test)
        print(f"[load] done: X_train {self.X_train.shape}, X_test {self.X_test.shape}")

    def convert_encoded_label_to_str(self, encoded_label):
        return self.label_encoder.inverse_transform(encoded_label)
