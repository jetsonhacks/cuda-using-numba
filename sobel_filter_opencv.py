import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter_ns

def sobel_filter(image_gray):

    # A Kernel size of 1 implies a simple 3x3 Sobel filter
    # Apply Sobel filter in X direction
    sobel_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=1)

    # Apply Sobel filter in Y direction
    sobel_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=1)

    # Normalize or clip the magnitude
    # √ (grad_x^2 + grad_y^2)        
    magnitude = np.hypot(sobel_x,sobel_y)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255
    # Convert magnitude to uint8
    return magnitude.astype(np.uint8)
  

def main():
   # Read the image using matplotlib
    print("Running sobel filter with OpenCV")
    # Load image
    image_path = 'images/MurderingSnowmen.jpeg'  # Update with your image path
    # Read the image in color
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale
    image_grayscale = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
 
    filtered_image = None
    def sobel_filter_helper():
        nonlocal filtered_image
        filtered_image = sobel_filter(image_grayscale)

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        sobel_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    times_to_run = 50
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