from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from lib.constants import CLASS_TO_IDX, PATH_TO_DATASET, RGB_BAND_INDICES

Sample = tuple[Path, int]


def discover_samples(root: Path = PATH_TO_DATASET) -> list[Sample]:
    """Walk the dataset folder and return every (tile_path, class_index) pair.

    The EuroSAT MS folder is laid out one subdirectory per class
    (e.g. ``EuroSAT_MS/Forest/Forest_1.tif``). For each class listed in
    ``CLASS_TO_IDX``, this collects every ``.tif`` file inside and pairs it
    with the integer label for that class. Iteration order is deterministic
    (classes follow ``CLASSES``, files are sorted alphabetically) so the
    same call always produces the same list — important for reproducible
    train/val/test splits.

    Args:
        root: Folder containing one subdirectory per class. Defaults to
            ``lib.constants.PATH_TO_DATASET``.

    Returns:
        List of ``(Path, label)`` tuples covering every .tif under ``root``.

    Raises:
        FileNotFoundError: A class subdirectory is missing.
        RuntimeError: No .tif files were found anywhere under ``root``.
    """

    root = Path(root)
    samples: list[Sample] = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for tif in sorted(class_dir.glob("*.tif")):
            samples.append((tif, label))
    if not samples:
        raise RuntimeError(f"No .tif files found under {root}")
    return samples


class EuroSATDataset(Dataset):
    """PyTorch ``Dataset`` over EuroSAT multispectral tiles.

    Each sample on disk is a 64×64 GeoTIFF with **13 spectral bands** from
    Sentinel-2 (see ``lib.constants.BANDS``). Unlike a normal RGB image
    (3 channels, 8-bit), every pixel here carries 13 reflectance values
    captured at different wavelengths — visible light, near-infrared,
    short-wave infrared, etc. That extra spectral information is what
    makes land-use classification on satellite imagery work well.

    What this class produces per sample:

    * ``image``: ``torch.float32`` tensor of shape ``(13, 64, 64)`` —
      the channel-first layout PyTorch convolutions expect (``C, H, W``).
      Values are the *raw* Sentinel-2 reflectance values cast to float32;
      no normalization or scaling is applied here. Normalization should
      live in the ``transform`` callable so train/val/test all use the
      same statistics computed in ``lib/``.
    * ``label``: ``int`` class index matching ``CLASS_TO_IDX``.

    Why a ``transform`` hook instead of normalizing inside ``__getitem__``:
    keeping the dataset raw means a single pass over the train split can
    compute per-band mean/std, and the *same* normalization function can
    then be passed to all three splits. It also lets the from-scratch
    model and the transfer-learning model use different transforms (e.g.
    the pretrained backbone may expect only the RGB bands resized to
    224×224) while sharing the same underlying file list.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        rgb_only: bool = False
    ):
        """
        Args:
            samples: Sequence of ``(path, label)`` pairs, typically a slice
                of ``discover_samples()`` produced by the split logic in
                ``lib.splits``.
            transform: Optional callable applied to each loaded float32
                tensor before it is returned. Use this for normalization,
                augmentation, resizing for a pretrained backbone, etc. When
                ``rgb_only`` is set, the transform sees a ``(3, 64, 64)``
                tensor; otherwise it sees ``(13, 64, 64)``.
            rgb_only: If True, keep only the natural-color RGB bands
                (R=B04, G=B03, B=B02 via ``RGB_BAND_INDICES``) so each
                sample is ``(3, 64, 64)`` in R, G, B order. This is what the
                transfer-learning model wants, since ImageNet-pretrained
                backbones expect 3-channel input. Band selection happens
                before ``transform``.
        """
        self.samples = list(samples)
        self.transform = transform
        self.rgb_only = rgb_only

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load one tile from disk and return ``(image_tensor, label)``.

        Steps:

        1. ``rasterio.open(path).read()`` returns a NumPy array of shape
           ``(bands, height, width)`` = ``(13, 64, 64)``. rasterio already
           gives us channel-first, which is exactly the layout PyTorch
           wants — no transpose needed.
        2. The on-disk dtype is ``uint16`` (Sentinel-2 reflectance is
           stored as 16-bit integers). We cast to ``float32`` because
           neural-network ops require floats, and ``torch.from_numpy``
           wraps the buffer as a tensor without copying when possible.
        3. If ``rgb_only`` is set, select the three natural-color bands
           (yielding a ``(3, 64, 64)`` tensor in R, G, B order) before any
           transform runs.
        4. If a ``transform`` was provided, it is applied to the tensor
           (e.g. per-band mean/std normalization, resizing for a
           pretrained backbone).
        """
        path, label = self.samples[idx]
        with rasterio.open(path) as src:
            img = src.read()  # shape: (13, 64, 64), dtype: uint16
        tensor = torch.from_numpy(img.astype(np.float32))
        if self.rgb_only:
            tensor = tensor[list(RGB_BAND_INDICES)]  # (3, 64, 64) in R, G, B order
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label
