import torch

def magnitude_max_pooling_1d(input_tensor, pool_size, stride):
    # Get the dimensions of the input tensor
    B, N, L = input_tensor.size()
    
    # Calculate the output length
    out_length = (L - pool_size) // stride + 1
    
    # Initialize the output tensor
    output_tensor = torch.zeros((B, N, out_length))
    
    # Unfold the input tensor to create sliding windows
    windows = input_tensor.unfold(2, pool_size, stride)
    
    # Reshape the windows to a 4D tensor
    windows = windows.contiguous().view(B, N, out_length, pool_size)
    
    # Compute the magnitudes of the values in each window
    magnitudes = torch.abs(windows)
    
    # Find the indices of the maximum magnitudes in each window
    max_indices = torch.argmax(magnitudes, dim=-1, keepdim=True)
    
    # Gather the values corresponding to the maximum magnitudes
    max_values = windows.gather(dim=-1, index=max_indices).squeeze(-1)
    
    return max_values

# Example usage
input_tensor = torch.tensor([[[1.0, -3.0, 2.0, 4.0, -1.0, 6.0, -7.0, 8.0, 5.0],
                              [9.0, -2.0, 7.0, -4.0, 3.0, -6.0, 5.0, -8.0, 4.0]]])
pool_size = 2
stride = 1
output_tensor = magnitude_max_pooling_1d(input_tensor, pool_size, stride)
print(output_tensor)
