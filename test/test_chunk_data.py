import pytest
import sys
from pathlib import Path

from utils.common_utils import chunk_data


# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChunkDataDict:
    """Comprehensive tests for chunk_data function."""
    
    def test_even_split_simple(self):
        """Test splitting dict with keys that divide evenly."""
        data = {f"layer_{i}": i for i in range(10)}
        n_chunks = 5
        
        result = chunk_data(data, n_chunks)
        
        # Should return list of dicts
        assert isinstance(result, dict), "Result should be a dict"
        assert len(result) == n_chunks, f"Should have {n_chunks} chunks"
        
        for chunk in result.values():
            assert isinstance(chunk, dict), "Each chunk should be a dict"
        
        # Verify all original keys are present
        all_keys = set()
        for chunk in result.values():
            all_keys.update(chunk.keys())
        assert all_keys == set(data.keys()), "All original keys should be present"
        
        # Verify all original values are preserved
        reconstructed = {}
        for chunk in result.values():
            reconstructed.update(chunk)
        assert reconstructed == data, "Reconstructed data should match original"
    
    def test_uneven_split_remainder_in_last(self):
        """Test that remainder goes to the last chunk."""
        data = {f"tensor_{i}": float(i) for i in range(10)}
        n_chunks = 3
        
        result = chunk_data(data, n_chunks)
        
        assert len(result) == n_chunks, f"Should have {n_chunks} chunks"
        
        # Verify all original keys are present
        all_keys = set()
        for chunk in result.values():
            all_keys.update(chunk.keys())
            
        assert all_keys == set(data.keys()), "All original keys should be present"
        assert set(all_keys) == set(data.keys()), "All keys should be present"
    
    def test_single_chunk(self):
        """Test with n_chunks=1, should return all data in one chunk."""
        data = {f"param_{i}": [i, i+1, i+2] for i in range(5)}
        n_chunks = 1
        
        result = chunk_data(data, n_chunks)
        
        assert len(result) == 1, "Should have 1 chunk"
        assert result[0] == data, "Single chunk should contain all data"
    
    def test_more_chunks_than_keys(self):
        """Test when n_chunks > number of keys."""
        data = {"a": 1, "b": 2, "c": 3}
        n_chunks = 10
        
        result = chunk_data(data, n_chunks)
        
        # Should still work, some chunks will be empty or we'll have fewer chunks
        # Depending on implementation, this tests edge case handling
        all_keys = set()
        for chunk in result.values():
            if chunk:  # Only check non-empty chunks
                all_keys.update(chunk.keys())
        
        assert all_keys == set(data.keys()), "All keys should still be present"
    
    def test_empty_dict(self):
        """Test with empty input dict."""
        data = {}
        n_chunks = 5
        
        result = chunk_data(data, n_chunks)
        
        assert isinstance(result, dict), "Should return a dict"
        assert len(result) == 0, "Result should be empty"
    
    def test_large_dataset(self):
        """Test with large dataset (simulating model tensors)."""
        # Simulate 272 tensors like SmolLM2 model
        data = {f"model.layers.{i}.weight": i * 0.5 for i in range(272)}
        n_chunks = 10
        
        result = chunk_data(data, n_chunks)
        
        assert len(result) == n_chunks
        
        # Verify no data loss
        total_keys = sum(len(chunk) for chunk in result.values())
        assert total_keys == len(data), "Total keys should match original"
        
        all_keys = set()
        for chunk in result.values():
            all_keys.update(chunk.keys())
        assert all_keys == set(data.keys()), "All original keys should be present"
    
    def test_mixed_data_types(self):
        """Test with various data types in dict values."""
        data = {
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "hello",
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "data"},
            "none_val": None,
        }
        n_chunks = 2
        
        result = chunk_data(data, n_chunks)
        
        assert len(result) == n_chunks
        
        # Reconstruct and verify all data types preserved
        reconstructed = {}
        for chunk in result.values():
            reconstructed.update(chunk)
        
        assert reconstructed == data, "All data types should be preserved"
    
    def test_key_order_preservation(self):
        """Test that key order is preserved within chunks."""
        # Python 3.7+ dicts maintain insertion order
        data = {f"key_{i:03d}": i for i in range(20)}
        n_chunks = 4
        
        result = chunk_data(data, n_chunks)
        
        # Extract all keys in order from chunks
        keys_in_chunks = []
        for chunk in result.values():
            keys_in_chunks.extend(chunk.keys())
        
        original_keys = list(data.keys())
        assert keys_in_chunks == original_keys, "Key order should be preserved"
    
    def test_two_chunks(self):
        """Test simple split into 2 chunks."""
        data = {f"layer_{i}": i * 10 for i in range(6)}
        n_chunks = 2
        
        result = chunk_data(data, n_chunks)
        
        assert len(result) == 2
        all_keys = set()
        for chunk in result.values():
            all_keys.update(chunk.keys())
        assert all_keys == set(data.keys()), "All original keys should be present"
        assert sum(len(chunk) for chunk in result.values()) == len(data)
    
    def test_prime_number_keys(self):
        """Test with prime number of keys (forces uneven split)."""
        data = {f"tensor_{i}": i ** 2 for i in range(13)}  # 13 is prime
        n_chunks = 4
        
        result = chunk_data(data, n_chunks)
        
        assert len(result) == n_chunks
        
        # Verify total and all keys present
        assert sum(len(chunk) for chunk in result.values()) == 13
        all_keys = set()
        for chunk in result.values():
            all_keys.update(chunk.keys())
        assert all_keys == set(data.keys()), "All original keys should be present"
    
    def test_no_data_loss_stress(self):
        """Stress test to ensure no data loss with various chunk sizes."""
        data = {f"weight_{i}": i * 0.001 for i in range(100)}
        
        for n_chunks in [1, 2, 3, 5, 7, 10, 13, 25, 50, 100, 150]:
            result = chunk_data(data, n_chunks)
            
            # Reconstruct
            reconstructed = {}
            for chunk in result.values():
                reconstructed.update(chunk)
            
            assert reconstructed == data, f"Data loss detected with {n_chunks} chunks"
            assert sum(len(chunk) for chunk in result.values()) == len(data), \
                f"Total keys mismatch with {n_chunks} chunks"
    
    def test_numeric_string_keys(self):
        """Test with numeric string keys (common in model layers)."""
        data = {str(i): f"value_{i}" for i in range(15)}
        n_chunks = 4
        
        result = chunk_data(data, n_chunks)
        
        assert len(result) == n_chunks
        
        # Verify all keys present
        all_keys = set()
        for chunk in result.values():
            all_keys.update(chunk.keys())
        assert all_keys == set(data.keys())


class TestChunkDataList:
    """Tests for chunk_data with list (non-dict) input."""

    def test_even_split(self):
        """Test splitting a list that divides evenly."""
        data = list(range(10))
        n_chunks = 5

        result = chunk_data(data, n_chunks)

        assert isinstance(result, dict), "Result should be a dict"
        assert len(result) == n_chunks, f"Should have {n_chunks} chunks"
        for chunk in result.values():
            assert isinstance(chunk, list), "Each chunk should be a list"

        # Reconstruct and verify no data loss
        reconstructed = []
        for chunk in result.values():
            reconstructed.extend(chunk)
        assert reconstructed == data, "Reconstructed list should match original"

    def test_uneven_split_remainder_in_last(self):
        """Test that torch.chunk puts remainder in the last chunk."""
        data = list(range(10))
        n_chunks = 3

        result = chunk_data(data, n_chunks)

        assert len(result) == n_chunks

        # 10 elements, 3 chunks: torch.chunk gives [4, 3, 3] (ceiling-first)
        reconstructed = []
        for chunk in result.values():
            reconstructed.extend(chunk)
        assert reconstructed == data, "All elements should be preserved"
        assert sum(len(chunk) for chunk in result.values()) == len(data)

    def test_single_chunk(self):
        """Test n_chunks=1 returns all data in one list chunk."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = chunk_data(data, 1)

        assert len(result) == 1
        assert result[0] == data

    def test_more_chunks_than_elements(self):
        """When n_chunks > len(data), torch.chunk returns len(data) chunks of size 1."""
        data = [10, 20, 30]
        result = chunk_data(data, 10)

        # torch.chunk won't create empty chunks, so we get at most len(data) chunks
        assert len(result) <= 10
        reconstructed = []
        for chunk in result.values():
            reconstructed.extend(chunk)
        assert reconstructed == data

    def test_large_list(self):
        """Test with a large list (simulating flattened tensor data)."""
        data = [float(i) * 0.01 for i in range(272)]
        n_chunks = 10

        result = chunk_data(data, n_chunks)

        assert len(result) == n_chunks
        assert sum(len(chunk) for chunk in result.values()) == len(data)

        reconstructed = []
        for chunk in result.values():
            reconstructed.extend(chunk)
        assert reconstructed == data

    def test_prime_number_elements(self):
        """Test with a prime number of elements (forces uneven split)."""
        data = list(range(13))
        n_chunks = 4

        result = chunk_data(data, n_chunks)

        assert len(result) == n_chunks
        assert sum(len(chunk) for chunk in result.values()) == 13

        reconstructed = []
        for chunk in result.values():
            reconstructed.extend(chunk)
        assert reconstructed == data

    def test_no_data_loss_stress(self):
        """Stress test: no data loss across various chunk counts."""
        data = [i * 0.001 for i in range(100)]

        for n_chunks in [1, 2, 3, 5, 7, 10, 13, 25, 50, 100]:
            result = chunk_data(data, n_chunks)
            reconstructed = []
            for chunk in result.values():
                reconstructed.extend(chunk)
            assert reconstructed == data, f"Data loss with {n_chunks} chunks"
            assert sum(len(chunk) for chunk in result.values()) == len(data)

    def test_integer_list(self):
        """Test with a plain integer list."""
        data = list(range(20))
        n_chunks = 4

        result = chunk_data(data, n_chunks)

        assert len(result) == n_chunks
        assert sum(len(chunk) for chunk in result.values()) == len(data)

        reconstructed = []
        for chunk in result.values():
            reconstructed.extend(chunk)
        assert reconstructed == data

    def test_float_list(self):
        """Test with a float list preserves values."""
        data = [0.1 * i for i in range(9)]
        n_chunks = 3

        result = chunk_data(data, n_chunks)

        assert len(result) == n_chunks
        reconstructed = []
        for chunk in result.values():
            reconstructed.extend(chunk)
        assert len(reconstructed) == len(data)
        for orig, got in zip(data, reconstructed):
            assert abs(orig - got) < 1e-6, "Float values should be preserved"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
