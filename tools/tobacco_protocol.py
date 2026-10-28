"""Canonical Tobacco-Clean16 v1 contract; no runtime dependencies."""

CLASSES = (
    'weather_fleck',
    'potato_virus_y',
    'tobacco_mosaic_virus',
    'cucumber_mosaic_virus',
    'alternaria_leaf_spot',
    'black_shank',
    'bacterial_wilt',
    'hollow_stalk',
    'wildfire',
    'potassium_deficiency',
    'magnesium_deficiency',
    'anthracnose',
    'frogeye_leaf_spot',
    'angular_leaf_spot',
    'sunburn',
    'herbicide_phytotoxicity',
)

MODEL_LABEL_COLUMNS = [
    'y00_c02_weather_fleck',
    'y01_c03_potato_virus_y',
    'y02_c04_tobacco_mosaic_virus',
    'y03_c05_cucumber_mosaic_virus',
    'y04_c06_alternaria_leaf_spot',
    'y05_c07_black_shank',
    'y06_c08_bacterial_wilt',
    'y07_c09_hollow_stalk',
    'y08_c11_wildfire',
    'y09_c12_potassium_deficiency',
    'y10_c13_magnesium_deficiency',
    'y11_c16_anthracnose',
    'y12_c17_frogeye_leaf_spot',
    'y13_c18_angular_leaf_spot',
    'y14_c19_sunburn',
    'y15_c20_herbicide_phytotoxicity',
]
NUM_CLASSES = len(CLASSES)
TASK_PROTOCOL = 'tobacco_clean16_v1'
