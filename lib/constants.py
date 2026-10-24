from pathlib import Path

# base folder of the eurosat dataset
PATH_TO_DATASET = (
    Path(__file__).resolve().parent.parent / "data" / "eurosat" / "EuroSAT_MS"
)

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Mapping of class name to index in CLASSES list constant
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

NUM_CLASSES = len(CLASSES)

# Band order in EuroSAT MS .tif files (Sentinel-2 spec)
BANDS = [
    "B01",
    "B02",  # Blue channel
    "B03",  # Green channel
    "B04",  # Red channel
    "B05",
    "B06",
    "B07",
    "B08",
    "B08A",
    "B09",
    "B10",
    "B11",
    "B12",
]
NUM_BANDS = len(BANDS)

# 0-indexed channel positions for natural-color RGB (R=B04, G=B03, B=B02)
RGB_BAND_INDICES = (3, 2, 1)

# Sentinel-2 surface reflectance is stored as uint16, but real-world values
# rarely exceed ~10000. Dividing by this constant maps typical reflectance into
# [0, 1] (occasional brighter pixels are clipped). This is the standard EuroSAT
# preprocessing convention, shared by every data loader in lib/ so the
# from-scratch and Keras models normalize identically.
S2_REFLECTANCE_SCALE = 10_000.0

SEED = 42
