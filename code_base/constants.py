CLASSES = [
    "kidney",
    "prostate",
    "largeintestine",
    "spleen",
    "lung",
    # Not target organ, just for pseudo
    # "liver"
]
PIXEL_SCALE = {
    "prostate": 6.26 / 0.4,
    "spleen": 0.4945 / 0.4,
    "lung": 0.7562 / 0.4,
    "kidney": 0.5 / 0.4,
    "largeintestine": 0.2290 / 0.4,
    # Not target organ, just for pseudo
    # "liver": 1.0,
}
PIXEL_SIZE = {
    "unscaled": 0.4,
    "prostate": 6.26,
    "spleen": 0.4945,
    "lung": 0.7562,
    "kidney": 0.5,
    "largeintestine": 0.2290,
    # Not target organ, just for pseudo
    # "liver": 0.4,
}
