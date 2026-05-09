import logging
import torch

logger = logging.getLogger(__name__)


def chunk_data(data, n_chunks: int = 10) -> dict:
    """Split data into chunks using PyTorch's chunk method."""
    
    data_chunks = {}
    
    assert n_chunks > 0, "n_chunks must be greater than 0"
    
    if isinstance(data, dict):
        # Indexing the data for chunking
        idx = torch.tensor(list(range(len(data))))
         
        # Chunk the tensor (automatically handles uneven divisions)
        chunked_tensors = torch.chunk(idx, n_chunks)
        
        # Convert back to dict format

        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            
            data_chunks[chunk_idx] = {k : v for item_idx, (k, v) in enumerate(data.items()) if item_idx in chunk_tensor}
            
    else:
        # Indexing the data for chunking
        idx = torch.tensor(list(range(len(data))))
        
        # Chunk the tensor (automatically handles uneven divisions)
        chunked_tensors = torch.chunk(idx, n_chunks)
        
        # Convert back to list format
        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            data_chunks[chunk_idx] = [data[chunk_idx] for chunk_idx in chunk_tensor]
        
    return data_chunks


def main():
    """Example usage of chunk_data."""
    data = {f"tensor_{i}": float(i) for i in range(10)}
    n_chunks = 3
    chunks = chunk_data(data, n_chunks)
    print(chunks)
    
    for i in range(0, 10, 4):
        print(i)
    
    
if __name__ == "__main__":
    main()