import os
import ffmpeg
import sys
import shutil

from time import sleep, time
from subprocess import check_call
from glob import glob


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
   WHITE  = '\33[37m'

IS_QUIET = True

NUMBER_STAGES = 20
CASCADE_TYPE = "LBP"
number_positives_images = number_negatives_images = 0

# sudo ln /dev/null /dev/raw1394


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

def create_positives_images(directory_models="trees", input_file="negatives/negatives.txt", directory_output="positives", isQuiet=False):
    stdout = None
    if isQuiet:
        stdout=open(os.devnull, 'wb')
    
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
                        "-h", "48"], stdout=stdout)
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

def convert_to_vec_file(directory="positives", isQuiet=False):
    stdout = None
    if isQuiet:
        stdout=open(os.devnull, 'wb')

    try:
        # Convert to a vec file
        check_call(["opencv_createsamples",
                        "-info", directory+"/positives.txt",
                        "-bg", directory+"/negatives.txt",
                        "-vec", directory+"/cropped.vec",
                        "-num", str(number_positives_images),
                        "-w", "48",
                        "-h", "48"], stdout=stdout)
        # -num 250 is the number of positives images
    except:
        print("Error: Cannot convert to vec file.")

def train_the_cascade(isQuiet=False):
    stdout = None
    if isQuiet:
        stdout=open(os.devnull, 'wb')

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
                        "-h", "48"], stdout=stdout)
    except:
        print("Error: Error while training the cascade.")
    os.chdir('..')

# Class that manage the elapsed time
class Duration:
    times = {}

    # Start a timer
    def start(self, name):
        Duration.times[name] = time()

    # End a timer (and store the result)
    def end(self, name):
        Duration.times[name] = time() - Duration.times[name]

    # End a timer (and store the result) and show the result
    def stop(self, name):
        Duration.times[name] = time() - Duration.times[name]
        print("Done. (in {}s)".format(round(Duration.times[name], 4)))

    # Show the result for one timer
    def show(self, name):
        print("{} done in {}s".format(name, round(Duration.times[name], 4)))

    # Show the result for every stored timer
    def showAll(self):
        print()
        print("All duration in seconds:")
        for key, value in Duration.times.items():
            print("{} => {}s".format(key, round(value, 4)))


#ex: python3 ./main.py
if len(sys.argv) == 1:
    print("{}Warning:{} By default the program execute everything, from the image extraction to the training".format(color.YELLOW, color.END))

    sleep(2)
    timer = Duration()

    print("Creating the folders...")
    timer.start("folders")
    createFolder('positives')
    createFolder('negatives')
    createFolder('classifiers')
    timer.end("folders")

    print("Converting the mp4 video into multiples jpg...")
    timer.start("mp4Conversion")
    convert_mp4_to_jpg(isQuiet=IS_QUIET)
    timer.end("mp4Conversion")

    print("Listing every negatives images into a file...")
    timer.start("listNegatives")
    list_every_negatives_images()
    timer.end("listNegatives")

    print("Creating the positives images...")
    timer.start("createPositives")
    create_positives_images(isQuiet=IS_QUIET)
    timer.end("createPositives")
    
    print("Combining all the positives image text files into one file...")
    timer.start("combinePositives")
    combine_all_positives_text_files()
    timer.end("combinePositives")

    print("Counting the images...")
    timer.start("countImages")
    count_images()
    timer.end("countImages")

    print("Converting the image into a vec file...")
    timer.start("convertToVec")
    convert_to_vec_file(isQuiet=IS_QUIET)
    timer.end("convertToVec")

    print("Training the cascade...")
    timer.start("trainTheCascade")
    train_the_cascade(isQuiet=IS_QUIET)
    timer.end("trainTheCascade")

    timer.showAll()
    print("Everything Done.")


#ex: python3 ./main.py createFolder
if len(sys.argv) == 2:
    execution_type = sys.argv[1] 
    if execution_type == "createFolders":
        timer = Duration()

        print("Creating the folders...")

        timer.start("folders")
        createFolder('positives')
        createFolder('negatives')
        createFolder('classifiers')

        timer.stop("folders")

    elif execution_type == "convertMp4ToJpg":
        timer = Duration()

        print("Converting the mp4 video into multiples jpg...")

        timer.start("mp4Conversion")
        convert_mp4_to_jpg(isQuiet=IS_QUIET)
        timer.stop("mp4Conversion")

    elif execution_type == "listEveryNegativesImages":
        timer = Duration()

        print("Listing every negatives images into a file...")

        timer.start("listNegatives")
        list_every_negatives_images()
        timer.stop("listNegatives")

    elif execution_type == "createPositivesImages":
        timer = Duration()

        print("Creating the positives images...")

        timer.start("createPositives")
        create_positives_images(isQuiet=IS_QUIET)
        timer.stop("createPositives")

    elif execution_type == "combineAllPositivesTextFiles":
        timer = Duration()

        print("Combining all the positives image text files into one file...")

        timer.start("combinePositives")
        combine_all_positives_text_files()
        timer.stop("combinePositives")

    elif execution_type == "convertToVecFile":
        timer = Duration()

        print("Counting the images...")

        timer.start("convertToVec")
        count_images()
        print("Converting the image into a vec file...")
        convert_to_vec_file(isQuiet=IS_QUIET)
        timer.stop("convertToVec")

    elif execution_type == "trainTheCascade":
        timer = Duration()

        print("Counting the images...")

        timer.start("trainTheCascade")
        count_images()
        print("Training the cascade...")
        train_the_cascade(isQuiet=IS_QUIET)
        timer.stop("trainTheCascade")


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