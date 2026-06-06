#!/usr/bin/env python3
"""
Python test implementation of tileRange and tile_range_kernel functions.
Single view version for testing corner cases.
"""

def tile_range_kernel_python(table_tileId, table_length, max_tileId, tile_range, index):
    """
    Python version of tile_range_kernel that uses index instead of CUDA block/thread.
    Single view version - matches CUDA code exactly.
    
    Args:
        table_tileId: list [table_length] - tile IDs for single view
        table_length: int - length of the table
        max_tileId: int - maximum tile ID
        tile_range: list [max_tileId+2] - output range array (initialized to -1)
        index: int - the index to process (replaces blockIdx.x * blockDim.x + threadIdx.x)
    """
    print(f"index: {index}")

    # head
    if index == 0:
        tile_id = table_tileId[index]
        tile_range[tile_id] = index
    
    # tail
    if index == table_length - 1:
        tile_range[max_tileId + 1] = table_length
    
    if index < table_length - 1:
        cur_tile = table_tileId[index]
        next_tile = table_tileId[index + 1]
        print(f"  cur_tile: {cur_tile}, next_tile: {next_tile}")
        print(f"  Before update: tile_range[{cur_tile}] = {tile_range[cur_tile]}, tile_range[{next_tile}] = {tile_range[next_tile]}")
        if cur_tile != next_tile:
            if cur_tile + 1 < next_tile:
                tile_range[cur_tile + 1] = index + 1
            tile_range[next_tile] = index + 1
        print(f"  After update: tile_range[{cur_tile}] = {tile_range[cur_tile]}, tile_range[{next_tile}] = {tile_range[next_tile]}")


def tile_range_kernel_python_fixed(table_tileId, table_length, max_tileId, tile_range, index):
    """
    Python version of tile_range_kernel that uses index instead of CUDA block/thread.
    Single view version - matches CUDA code exactly.
    
    Args:
        table_tileId: list [table_length] - tile IDs for single view
        table_length: int - length of the table
        max_tileId: int - maximum tile ID
        tile_range: list [max_tileId+2] - output range array (initialized to -1)
        index: int - the index to process (replaces blockIdx.x * blockDim.x + threadIdx.x)
    """
    print(f"index: {index}")

    # head
    if index == 0:
        tile_id = table_tileId[index]
        tile_range[tile_id] = index
    
    # tail
    if index == table_length - 1:
        tile_range[max_tileId + 1] = table_length
        # The fix at tail
        cur_tile = table_tileId[index]
        tile_range[cur_tile+1] = table_length
    
    if index < table_length - 1:
        cur_tile = table_tileId[index]
        next_tile = table_tileId[index + 1]
        print(f"  cur_tile: {cur_tile}, next_tile: {next_tile}")
        print(f"  Before update: tile_range[{cur_tile}] = {tile_range[cur_tile]}, tile_range[{next_tile}] = {tile_range[next_tile]}")
        if cur_tile != next_tile:
            if cur_tile + 1 < next_tile:
                tile_range[cur_tile + 1] = index + 1
            tile_range[next_tile] = index + 1
        print(f"  After update: tile_range[{cur_tile}] = {tile_range[cur_tile]}, tile_range[{next_tile}] = {tile_range[next_tile]}")


def tileRange_python(table_tileId, table_length, max_tileId, use_fixed=False):
    """
    Python version of tileRange function for single view.
    
    Args:
        table_tileId: list [table_length] - sorted tile IDs
        table_length: int - length of the table
        max_tileId: int - maximum tile ID
        use_fixed: bool - whether to use the fixed version of the kernel
        
    Returns:
        list [max_tileId+2] - tile range array where tile_range[t] 
        indicates the starting index in the sorted table for tile t
    """
    # Initialize tile_range with -1 values as requested
    tile_range = [-1 for _ in range(max_tileId + 2)]
    
    # Process each index sequentially (simulating parallel CUDA threads)
    kernel_func = tile_range_kernel_python_fixed if use_fixed else tile_range_kernel_python
    for index in range(table_length):
        kernel_func(table_tileId, table_length, max_tileId, tile_range, index)
    
    return tile_range


def compute_gaussian_count_per_tile(tile_start_index: list) -> list:
    """
    Compute the number of Gaussians per tile from tile start indices.
    Python version for testing.
    
    Args:
        tile_start_index: List of length [total_tiles + 2] containing the starting 
                         index of Gaussians for each tile. Note that tile_id 0 is invalid,
                         so valid tiles are indexed from 1 to total_tiles.
    
    Returns:
        gaussian_count_per_tile: List of length [total_tiles] with number of Gaussians per tile
    
    Note:
        The gaussian_count_per_tile is 0-based, so gaussian_count_per_tile[i] represents 
        the Gaussian count for tile i+1 (since tile_id 0 is invalid).
        
        total_tiles is derived from len(tile_start_index) - 2 since tile_start_index 
        is guaranteed to have length total_tiles + 2.
    """
    # Derive total_tiles from the length of tile_start_index
    total_tiles = len(tile_start_index) - 2
    
    # Extract start indices for valid tiles (1 to total_tiles)
    start_indices = tile_start_index[1:total_tiles+1]  # [total_tiles]
    
    # Extract end indices (start of next tile) for valid tiles
    end_indices = tile_start_index[2:total_tiles+2]    # [total_tiles]
    
    # Do subtraction immediately
    gaussian_count_per_tile = [end - start for end, start in zip(end_indices, start_indices)]
    
    # Handle special cases: if start_indices is -1, then gaussian_count_per_tile is 0
    gaussian_count_per_tile = [0 if start == -1 else count for start, count in zip(start_indices, gaussian_count_per_tile)]
    
    # Handle special cases: if end_indices is -1, then gaussian_count_per_tile is 0
    gaussian_count_per_tile = [0 if end == -1 else count for end, count in zip(end_indices, gaussian_count_per_tile)]
    
    return gaussian_count_per_tile


def test_case(case_num, table_tileId, table_length, max_tileId, use_fixed=False):
    """Test a case and print formatted output."""
    version_label = "FIXED" if use_fixed else "BUGGY"
    print(f"=== Case {case_num} ({version_label}) ===")
    
    tile_range = tileRange_python(table_tileId, table_length, max_tileId, use_fixed)
    print(f"table_length: {table_length}, max_tileId: {max_tileId}")
    
    # Print table_tileId with indices
    indices_formatted = [f"{i:2d}" for i in range(len(table_tileId))]
    table_tileId_formatted = [f"{x:2d}" for x in table_tileId]
    print(f"indices:      [{', '.join(indices_formatted)}]")
    print(f"table_tileId: [{', '.join(table_tileId_formatted)}]")
    
    # Print tile_range with tile IDs
    tile_ids = [f"{i:2d}" for i in range(len(tile_range))]
    tile_range_formatted = [f"{x:2d}" if x != -1 else "-1" for x in tile_range]
    print(f"tile_ids:     [{', '.join(tile_ids)}]")
    print(f"tile_range:   [{', '.join(tile_range_formatted)}]")
    
    # Calculate and print gaussian count per tile
    gaussian_count = compute_gaussian_count_per_tile(tile_range)
    count_tile_ids = [f"{i+1:2d}" for i in range(len(gaussian_count))]  # +1 because tile IDs start from 1
    count_formatted = [f"{x:2d}" for x in gaussian_count]
    print(f"count_tiles:  [{', '.join(count_tile_ids)}]")
    print(f"gauss_count:  [{', '.join(count_formatted)}]")
    print()


if __name__ == "__main__":
    # This is a normal case, and the CUDA function works correctly
    test_case(1, [1, 2, 2, 3, 3, 3], 6, 3, use_fixed=False)
    test_case(1, [1, 2, 2, 3, 3, 3], 6, 3, use_fixed=True)

    # This is a corner case that the CUDA function fails to handle correctly
    # E.g., tile id 11
    test_case(2, [4, 4, 4, 8, 8, 11, 11, 11, 11], 9, 13, use_fixed=False)
    test_case(2, [4, 4, 4, 8, 8, 11, 11, 11, 11], 9, 13, use_fixed=True)

    # This is a corner case that the CUDA function fails to handle correctly
    # E.g., tile id 11
    test_case(3, [11, 11, 11, 11], 4, 13, use_fixed=False)
    test_case(3, [11, 11, 11, 11], 4, 13, use_fixed=True)