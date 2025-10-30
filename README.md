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

todo

## lab 3

## lab 4

