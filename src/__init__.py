"""RayScale ML Platform - End-to-end distributed ML platform using Ray."""

__version__ = "0.1.0"
__author__ = "Bhaskar Boruah"
__email__ = "boruah.bhaskar@gmail.com"

from src.config import settings
from src.utils.logging import configure_logging

# Configure logging on import
configure_logging()
