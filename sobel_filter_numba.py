import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import jit, prange, uint8, float64, float32
from time import perf_counter_ns

@jit(uint8[:, :](float32[:, :], float32[:, :]), nopython=True)
def compute_magnitude(sobel_x, sobel_y):
    # Initialize the magnitude array
    magnitude = np.empty_like(sobel_x)

    # Compute the magnitude
    for i in range(sobel_x.shape[0]):
        for j in range(sobel_x.shape[1]):
            magnitude[i, j] = min(255, np.sqrt(sobel_x[i, j]**2 + sobel_y[i, j]**2))
    
    return magnitude.astype(np.uint8)  


@jit(uint8[:, :](float32[:, :]), nopython=True, parallel=True, )
def sobel_filter_parallel(input_image):
    """
    Apply a Sobel filter to the image.
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Initialize the gradient images
    sobel_x = np.zeros_like(input_image)
    sobel_y = np.zeros_like(input_image)

    # Apply the filter
    # Parallelize the outer loop
    for i in prange(1, input_image.shape[0] - 1):
        for j in prange(1, input_image.shape[1] - 1):
            sobel_x[i, j] = np.sum(Gx * input_image[i-1:i+2, j-1:j+2])
            sobel_y[i, j] = np.sum(Gy * input_image[i-1:i+2, j-1:j+2])

    magnitude = np.hypot(sobel_x, sobel_y)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255
    return magnitude.astype(np.uint8)  # Convert to uint8 if necessary

    # Initialize the magnitude array
    magnitude = np.empty_like(sobel_x, dtype=np.uint8)

    # Compute the magnitude in parallel
    for i in prange(input_image.shape[0]):
        for j in prange(input_image.shape[1]):
            mag = np.sqrt(sobel_x[i, j]**2 + sobel_y[i, j]**2)
            # Manually implement clipping
            magnitude[i, j] = min(255, max(0, mag))

    return magnitude
 
  
# The input_image should be grayscale
@jit(nopython=True)
def sobel_filter(input_image):
    """
    Apply a Sobel filter to the image.
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Initialize the gradient images
    grad_x = np.zeros_like(input_image)
    grad_y = np.zeros_like(input_image)

    # Apply the filter
    for i in range(1, input_image.shape[0] - 1):
        for j in range(1, input_image.shape[1] - 1):
            grad_x[i, j] = np.sum(Gx * input_image[i-1:i+2, j-1:j+2])
            grad_y[i, j] = np.sum(Gy * input_image[i-1:i+2, j-1:j+2])

    magnitude = np.hypot(grad_x,grad_y)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255
    return magnitude.astype(np.uint8)  # Convert to uint8 if necessary

def main():
    # Read the image using matplotlib
    print("Running sobel filter using numba jit")
    # Load image
    image_path = 'images/MurderingSnowmen.jpeg'  # Update with your image path
    input_image = mpimg.imread(image_path)

    # Convert to grayscale if the image is in color
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])

    input_image = input_image.astype(np.float32)
    # Apply Sobel filter
    filtered_image = None
    def sobel_filter_helper():
        nonlocal filtered_image
        filtered_image = sobel_filter(input_image)

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        sobel_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    times_to_run = 10
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        sobel_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 


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