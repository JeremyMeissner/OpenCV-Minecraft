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


# https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


img = cv2.imread("image_test.png") 

#OpenCV opens images as BRG  
# but we want it as RGB and  
# we also need a grayscale  
# version 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 



# Use minSize because for not  
# bothering with extra-small  
# dots that would look like STOP signs 
stop_data = cv2.CascadeClassifier('cascade.xml') 

found = stop_data.detectMultiScale(img_gray, minSize =(20, 20))

# Don't do anything if there's  
# no sign 
amount_found = len(found) 
  
  
if amount_found != 0: 
      
    # There may be more than one 
    # sign in the image 
    for (x, y, width, height) in found: 
          
        # We draw a green rectangle around 
        # every recognized sign 
        cv2.rectangle(img_rgb, (x, y),  
                      (x + height, y + width),  
                      (0, 255, 0), 5) 


# Creates the environment  
# of the picture and shows it 
plt.subplot(1, 1, 1) 
plt.imshow(img_rgb) 
plt.show() 

