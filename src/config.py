from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SEED = 42
TEST_SIZE = 0.2
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DROP_COLS = ["UDI", "Product ID"] + FAILURE_MODES