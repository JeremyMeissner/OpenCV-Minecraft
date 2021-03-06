import os
import cv2
import matplotlib.pyplot as plt

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

def detect_something_in_image(input_file, trees=["oak_tree"]):
    # Open the tested image
    img = cv2.imread(input_file) 

    # Open the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # Convert image to a gray one because the model was train in grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # I've done this because I wanted to add more type in the futur so, I'll just need to add to the array
    for tree_type in trees:

        # Import the file training model into OpenCV
        oak_data = cv2.CascadeClassifier('models/{}.xml'.format(tree_type)) 

        # Detect all the tree on the image (thanks to the training model)
        found = oak_data.detectMultiScale(img_gray)

        # Draw something only if there is more than 0 tree
        if len(found) != 0: 
            
            # If there are more tree in the image
            for (x, y, width, height) in found: 
                
                # Draw a green rectangle
                cv2.rectangle(img_rgb, (x, y),  
                            (x + height, y + width),  
                            (0, 255, 0), 5) 
                
                # The text height is needed because I want to put the text under the rectangle and I need to substract itself to go under the rectangle
                ((text_width, txt_height), o) = cv2.getTextSize(tree_type, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)

                # Draw a title next to the green rectangle
                cv2.putText(img_rgb, tree_type, (x, y + height + txt_height), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

        # Create the final image with the rectangle and the text
        plt.subplot(1, 1, 1) 
        plt.imshow(img_rgb) 
        plt.show() 

detect_something_in_image("image_test2.png")


