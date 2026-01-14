from pathlib import Path
import sys
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session):
    """Runs once before all tests. Bootstraps the test data."""
    # Ensure the code can be imported even if running from inside /tests
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Import the sampling script after updating sys.path
    from scripts.sample import create_sample_data

    sample_path = PROJECT_ROOT / "tests" / "sample_data.pt"

    if not sample_path.exists():
        logger.debug(f"Generating missing test data at {sample_path}")
        create_sample_data()
