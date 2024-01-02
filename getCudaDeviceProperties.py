from numba import cuda

# Get the current CUDA device
device = cuda.get_current_device()

# Query device properties
max_threads_per_block = device.MAX_THREADS_PER_BLOCK
max_block_dim_x = device.MAX_BLOCK_DIM_X
max_block_dim_y = device.MAX_BLOCK_DIM_Y
max_block_dim_z = device.MAX_BLOCK_DIM_Z
max_grid_dim_x = device.MAX_GRID_DIM_X
max_grid_dim_y = device.MAX_GRID_DIM_Y
max_grid_dim_z = device.MAX_GRID_DIM_Z

# Print the properties
print("Max threads per block:", max_threads_per_block)
print("Max block dimensions (x, y, z):", max_block_dim_x, max_block_dim_y, max_block_dim_z)
print("Max grid dimensions (x, y, z):", max_grid_dim_x, max_grid_dim_y, max_grid_dim_z)
