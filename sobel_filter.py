import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import perf_counter_ns

# The input image should be grayscale
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

    # Normalize or clip the magnitude
    magnitude = np.hypot(grad_x,grad_y)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255
    return magnitude.astype(np.uint8)  # Convert to uint8 if necessary

def main():
    # Read the image using matplotlib
    print("Running sobel filter")
    # Load image
    image_path = 'images/MurderingSnowmen.jpeg'  # Update with your image path
    input_image = mpimg.imread(image_path)

    # Convert to grayscale if the image is in color
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])

    input_image = input_image.astype(np.float32)
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