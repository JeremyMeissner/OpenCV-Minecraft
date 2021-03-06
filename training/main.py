import os
import ffmpeg
import cv2
import matplotlib.pyplot as plt
import shutil

from time import sleep
from subprocess import check_call
from glob import glob

"""
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   CWHITE  = '\33[37m'
"""

NUMBER_STAGES = 20
CASCADE_TYPE = "LBP"
number_positives_images = number_negatives_images = 0

# https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
# Purpose: Create a folder is it doesn't already exist
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# https://stackoverflow.com/q/845058/11091778
# Purpose: Count the number of line in a file, fast
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def deleteFolder(directory):
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    except OSError:
        print ('Error: Cannot delete directory. ' +  directory)

def convert_mp4_to_jpg(file_input="trees/input.mp4", directory_output="negatives", fps="3", isQuiet=False):
    # Convert the mp4 input into multiple images (3 images for every seconds of the video)
    ffmpeg.input(file_input).filter('fps', fps=str(fps)).output(directory_output + '/out%d.jpg', start_number=0, **{'qscale:v': 2}).overwrite_output().run(quiet=isQuiet)


def list_every_negatives_images(directory="negatives"):
    # Alternative of ls (negativeImage is the folder where there are all the images proccess above)
    negative_images = glob(directory+'/*.jpg')

    # Reference all the images into a text file
    try:
        with open(directory+"/negatives.txt", "w+") as txt_file:
            for line in negative_images:
                txt_file.write("{}\n".format(os.path.basename(line)))
    except:
        print("Error: Cannot create text file that reference negatives images.")

def create_positives_images(directory_models="trees", input_file="negatives/negatives.txt", directory_output="positives"):
    # Alternative of ls (trees is the folder where there are all the trees image)
    original_images = glob(directory_models+"/*.jpg")
    
    # Create the positives images for each trees images
    try:
        x = 0
        for image_name in original_images:
            check_call(["opencv_createsamples",
                        "-img", directory_models+"/{}".format(os.path.basename(image_name)),
                        "-bg", input_file,
                        "-info", directory_output+"/cropped{}.txt".format(x),
                        "-num", "128",
                        "-maxxangle", "0.0",
                        "-maxyangle", "0.0",
                        "-maxzangle", "0.3",
                        "-bgcolor", "255",
                        "-bgthresh", "8",
                        "-w", "48",
                        "-h", "48"])
            x += 1
    except:
        print("Error: Cannot create positives images.")

def combine_all_positives_text_files(directory="positives"):
    lines = []
    # Alternative of ls (get all the txt file in the folder)
    positives_text_files = glob(directory+"/*.txt")

    # For each file selected above, put each line in an array (lines)
    for files in positives_text_files:
        temp_file = open(files, "r")
        for line in temp_file:
            lines.append(line)

    # Put every line of the array (lines) into an unique file
    with open(directory+"/positives.txt", "w+") as txt_file:
        for line in lines:
            txt_file.write("{}".format(line))

def count_images():
    global number_positives_images, number_negatives_images
    # Count the number of positives and negatives images
    number_positives_images = file_len("positives/positives.txt")
    number_negatives_images = file_len("negatives/negatives.txt")

def convert_to_vec_file(directory="positives"):
    try:
        # Convert to a vec file
        check_call(["opencv_createsamples",
                        "-info", directory+"/positives.txt",
                        "-bg", directory+"/negatives.txt",
                        "-vec", directory+"/cropped.vec",
                        "-num", str(number_positives_images),
                        "-w", "48",
                        "-h", "48"])
        # -num 250 is the number of positives images
    except:
        print("Error: Cannot convert to vec file.")

def train_the_cascade():
    # Alternative of cd (needed because the negatives.txt file only have the file name)
    os.chdir('negatives')
    # Train the Haar cascade 
    try:
        check_call(["opencv_traincascade",
                        "-data", "../classifiers",
                        "-vec", "../positives/cropped.vec",
                        "-bg", "negatives.txt",
                        "-numPos", str(number_positives_images - (10*number_positives_images/100)), # Remove 10% because opencv_traincascade can take a bit more images
                        "-numNeg", str(number_negatives_images - (10*number_negatives_images/100)),
                        "-numStages", str(NUMBER_STAGES),
                        "-precalcValBufSize", "1024",
                        "-precalcIdxBufSize", "1024",
                        "-featureType", CASCADE_TYPE,
                        "-minHitRate", "0.999",
                        "-maxFalseAlarmRate", "0.5",
                        "-mode", "ALL",
                        "-w", "48",
                        "-h", "48"])
    except:
        print("Error: Error while training the cascade.")
    os.chdir('..')


createFolder('positives')
createFolder('negatives')
createFolder('classifiers')
convert_mp4_to_jpg()
list_every_negatives_images()
create_positives_images()
combine_all_positives_text_files()
count_images()
convert_to_vec_file()
train_the_cascade()



##############
# UNIT TESTS #
##############
"""
def test_convert_mp4_to_jpg():
    # Delete folder if exist and create a new one
    deleteFolder("test_negatives")
    createFolder("test_negatives")
    # Call the function
    convert_mp4_to_jpg("test/input.mp4", "test_negatives", 1, True)
    # Test the length of the nubmer of 
    assert len(glob('test_negatives/out*.jpg'))
    deleteFolder("test_negatives")

def test_list_every_negatives_images():
    for f in glob("test/0*.jpg"):
        os.remove(f)
    for f2 in glob("test/cropped*.txt"):
        os.remove(f2)
    list_every_negatives_images("test")
    assert os.path.exists("test/negatives.txt")

def test_create_positives_images():
    create_positives_images("test", "test/negatives.txt", "test")
    assert len(glob('test/cropped*.txt'))


def test_combine_all_positives_text_files():
    os.remove("test/positives.txt")
    combine_all_positives_text_files("test")
    assert os.path.exists("test/positives.txt")

def test_count_images():
    count_images()
    assert number_positives_images+number_negatives_images

def test_convert_to_vec_file():
    convert_to_vec_file("test")
    assert os.path.exists("test/cropped.vec")
    os.remove("test/cropped.vec")
"""