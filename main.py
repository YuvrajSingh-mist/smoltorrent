import threading
import socket
import mlx.core as mx
import logging
import yaml

from utils.common_utils import chunk_data

logger = logging.getLogger(__name__)

with open("configs/config.yaml", "r") as f:
    config = yaml. safe_load(f)

def load_data(file_path: str) -> dict:
    """Load data from a file."""
    data = mx.load(file_path)
    return data



def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting SmolTorrent...")
    
    logger.info("Loading data...")
    data = load_data(config["data_path"])
    total_bytes = sum(v.nbytes for v in data.values())
    logger.info(f"Loaded {len(data)} tensors, total size {total_bytes / 1024**2:.1f} MB")
    
    chunked_data = chunk_data(data, chunk_size_mb=10)  # 10 MB chunks
    logger.info(f"Split data into {len(chunked_data)} chunks")
    
    
    
    

if __name__ == "__main__":
    main()
