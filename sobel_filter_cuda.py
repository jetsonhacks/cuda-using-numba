from numba import cuda, void, float32, uint8, int32
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from time import perf_counter_ns

@cuda.jit(void(float32[:,:], uint8[:, :], int32[:, :], int32[:, :]))
def sobel_filter(input_image, output_image, kernel_x, kernel_y):
    """
    Apply a Sobel filter to the image using CUDA with variable-sized kernel arrays.
    """
    i, j = cuda.grid(2)
    height, width = input_image.shape
    kernel_height, kernel_width = kernel_x.shape

    # Calculate the margins to avoid boundary issues
    margin_y, margin_x = kernel_height // 2, kernel_width // 2

    if (margin_y <= i < height - margin_y) and (margin_x <= j < width - margin_x):
        Gx, Gy = 0, 0

        # Apply Sobel operator dynamically based on kernel size
        for u in range(-margin_y, margin_y + 1):
            for v in range(-margin_x, margin_x + 1):
                img_val = input_image[i + u, j + v]
                Gx += kernel_x[u + margin_y, v + margin_x] * img_val
                Gy += kernel_y[u + margin_y, v + margin_x] * img_val

        # The magnitude of the gradient
        output_image[i, j] = min(255, max(0, math.sqrt(Gx**2 + Gy**2)))
        # Approximated magnitude of the gradient
        # magnitude = abs(Gx) + abs(Gy)
        # Clamping the output to the range [0, 255]
        # output_image[i, j] = min(255, magnitude)

def main():
    # Read the image using matplotlib
    print("Running sobel filter using cuda")
    # Load image
    image_path = 'images/MurderingSnowmen.jpeg'  # Update with your image path
    input_image = mpimg.imread(image_path)

    # Check if the image is in color (3 channels)
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        # Convert to grayscale using np.dot
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])
 
    input_image = input_image.astype(np.float32)

    # Define block size and grid size
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(input_image.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(input_image.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Define Sobel kernels
    SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # These are constants, put them on the device for fast access
    d_sobel_x = cuda.to_device(SOBEL_X)
    d_sobel_y = cuda.to_device(SOBEL_Y)
    filtered_image = None

    def sobel_filter_helper():
        nonlocal filtered_image
        input_image_device = cuda.to_device(input_image)
        result_device = cuda.device_array(input_image.shape, np.uint8)
        sobel_filter[blockspergrid, threadsperblock](input_image_device, result_device, d_sobel_x, d_sobel_y)
        # Copy the result back to the host
        filtered_image = result_device.copy_to_host()

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        sobel_filter_helper()
        cuda.synchronize()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    times_to_run = 1000
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        sobel_filter_helper()
        cuda.synchronize()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    del d_sobel_x
    del d_sobel_y

    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Sobel Filter Result')
    plt.axis('off')
    plt.show()


if __name__ == '__main__' :
    main()