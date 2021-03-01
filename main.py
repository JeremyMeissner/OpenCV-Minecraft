import os
import ffmpeg
import cv2
import matplotlib.pyplot as plt

from time import sleep
from subprocess import check_call
from glob import glob
"""
final_file = []
username = sys.argv[1]
wordlist = sys.argv[2]
select_os = sys.argv[3]
number_of_worker = int(sys.argv[4])
worker_id = int(sys.argv[5])
filename = wordlist + '.txt'


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

def convert_mp4_to_jpg():
    # Convert the mp4 input into multiple images (3 images every seconds)
    ffmpeg.input('input.mp4').filter('fps', fps='3').output('./negativeImage/out%d.jpg', start_number=0, **{'qscale:v': 2}).overwrite_output().run() #quiet=True


def list_every_negatives_images():
    # Alternative of ls (negativeImage is the folder where there are all the images proccess above)
    negative_images = os.listdir('negativeImage')

    # Reference all the images into a text file
    with open("negativeImage/negatives.txt", "w+") as txt_file:
        for line in negative_images:
            txt_file.write("{}\n".format(line))

def create_positives_images():
    # Alternative of ls (trees is the folder where there are all the trees image)
    original_images = os.listdir('trees')
    x = 0

    # Create the positives images for each trees images
    for image_name in original_images:
        check_call(["opencv_createsamples",
                    "-img", "trees/{}".format(image_name),
                    "-bg", "negativeImage/negatives.txt",
                    "-info", "sampleImageTest/cropped{}.txt".format(x),
                    "-num", "128",
                    "-maxxangle", "0.0",
                    "-maxyangle", "0.0",
                    "-maxzangle", "0.3",
                    "-bgcolor", "255",
                    "-bgthresh", "8",
                    "-w", "48",
                    "-h", "48"])
        x += 1

def combine_all_positives_text_files():
    lines = []
    # Alternative of ls (get all the txt file in the folder)
    positives_text_files = glob("sampleImageTest/*.txt")

    # For each file selected above, put each line in an array (lines)
    for files in positives_text_files:
        temp_file = open(files, "r")
        for line in temp_file:
            lines.append(line)

    # Put every line of the array (lines) into an unique file
    with open("sampleImageTest/positives.txt", "w+") as txt_file:
        for line in lines:
            txt_file.write("{}".format(line))

def convert_to_vec_file():
    # Convert to a vec file
    check_call(["opencv_createsamples",
                    "-info", "sampleImageTest/positives.txt",
                    "-bg", "negativeImageDirectory/negatives.txt",
                    "-vec", "cropped.vec",
                    "-num", "250",
                    "-w", "48",
                    "-h", "48"])
    # -num 250 is the number of positives images

def train_the_cascade():
    # Alternative of cd (needed because the negatives.txt file only have the file name)
    os.chdir('negativeImage')
    # Train the Haar cascade 
    check_call(["opencv_traincascade",
                    "-data", "../classifier",
                    "-vec", "../cropped.vec",
                    "-bg", "negatives.txt",
                    "-numPos", "200",
                    "-numNeg", "600",
                    "-numStages", "10",
                    "-precalcValBufSize", "1024",
                    "-precalcIdxBufSize", "1024",
                    "-featureType", "HAAR",
                    "-minHitRate", "0.995",
                    "-maxFalseAlarmRate", "0.5",
                    "-w", "48",
                    "-h", "48"])
    os.chdir('..')


train_the_cascade()

