from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from satellite_image_classification.config import CLASS_MAPPING, NUM_CLASSES


def test_class_mapping_size():
    assert len(CLASS_MAPPING) == NUM_CLASSES
    assert NUM_CLASSES == 4

