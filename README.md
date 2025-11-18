# BIOMOD-Image-analysis-notes
Image analysis labs done in BIOMOD, with my annotations.

## For reference to create virtual environments:

python3 -m venv env
source env/bin/activate <br>
deactivate -> to leave the virtual environment

Installing packages:
pip list → shows the packages that are already installed <br>
pip install packagename

For Future Uses:
If the first time, put all packages needed into a txt file: <br>
pip freeze > requirements.txt

Now others can easily install all packages needed at once, with the text file: <br>
pip install -r requirements.txt

## lab 1

Lab 1 was a warm-up to openCV functions and manipulating images. <br>
excercises included:
- image rotation
- image reflection
- manipulating color values
<br>

### important functions/explanations:

- .shape -> function from the numpy library that takes in an image and returns its dimensions (height, width, channels)
- .dtype -> returns the data type that the array contains

#### showing/plotting an image:
def show_image(img, title='', cmap=None):
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

- cmap is part of the matplotlib library that maps numerical values to colors (RGB).




## lab 2

Created a grayscale histogram, which shows the pixel intensity distributions of an image. We can **equalize** that histogram by spreading out the most frequent intensity levels, so there will be a larger variety of intensities in the image. 

vocab:

Convolution - Convolution is a mathematical operation where a kernal slides incrementally over an image 
to combine values of all the pixels close by, which can create effects such as blurring, sharpening or edge detection.
Kernel - A kernel is a small matrix that is used for convolution. It is designed to define
a pixels relationship with its neighbours, which is outputed using the dot product.

Gaussian Blur - Gaussian blur smooths the image by averaging pixels with their neighbors, weighted by a Gaussian kernel.

Sobel Edge Detection - The Sobel operator detects edges by calculating gradients in the X and Y directions.


## lab 3
todo

## lab 4
todo

