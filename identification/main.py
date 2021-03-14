import os
import sys
import cv2
import matplotlib.pyplot as plt

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


VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
TREE_LIST = ["oak_tree", "birch_tree"]

#ex: python3 ./main.py
if len(sys.argv) == 1:
    print("{}Error:{} Please specify a detection type <image|video> <image_source|video_id> <tree_list>".format(color.RED, color.END))
    sys.exit()

detection_type = sys.argv[1]

#ex: python3 ./main.py "image"
if len(sys.argv) == 2:
    if detection_type == "image":
        print("{}Error:{} Please specify a input source ex: img/text.png".format(color.RED, color.END))
        sys.exit()
    if detection_type == "video":
        input_element = 0

#ex: python3 ./main.py "image" "salut.png"
if len(sys.argv) == 3:
    if detection_type == "image":
        # Check if the given file exist
        if not os.path.exists(sys.argv[2]):
            print("{}Error:{} The input file image doesnt exist".format(color.RED, color.END))
            sys.exit()
        else:
            input_element = sys.argv[2]
    if detection_type == "video":
        input_element = int(sys.argv[2])

if len(sys.argv) > 3:
    print("{}Error:{} Too much arguments, excepted: 2, given: {}".format(color.RED, color.END, len(sys.argv)-1))
    sys.exit()


def detect_something_in_image(input_file, trees):
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
                ((text_width, text_height), o) = cv2.getTextSize(tree_type, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)

                # Draw a title next to the green rectangle
                cv2.putText(img_rgb, tree_type, (x, y + height + text_height), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

        # Create the final image with the rectangle and the text
        plt.subplot(1, 1, 1) 
        plt.imshow(img_rgb) 
        plt.show() 

def detect_something_in_stream(video_id, trees):
    data = []

    # Open the stream video flux
    video = cv2.VideoCapture(video_id) 

    # Define the video size (VIDEO_WIDTH x VIDEO_HEIGHT)
    video.set(3, VIDEO_WIDTH)
    video.set(4, VIDEO_HEIGHT)

    # I've done this because I wanted to add more type in the futur so, I'll just need to add to the array
    for i, tree_type in enumerate(trees):
        # Import the file training model into OpenCV
        data.append(cv2.CascadeClassifier('models/{}.xml'.format(tree_type)))

    while True:
        # Select a frame from the stream video flux
        success, img = video.read()

        # Open the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        # Convert image to a gray one because the model was train in grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # I've done this because I wanted to add more type in the futur so, I'll just need to add to the array
        for i, tree_type in enumerate(trees):

            # Detect all the tree on the image (thanks to the training model)
            found = data[i].detectMultiScale(img_gray)

            # Draw something only if there is more than 0 tree
            if len(found) != 0: 
                
                # If there are more tree in the image
                for (x, y, width, height) in found: 
                    
                    # Draw a green rectangle
                    cv2.rectangle(img_rgb, (x, y),  
                                (x + height, y + width),  
                                (0, 255, 0), 5) 
                    
                    # The text height is needed because I want to put the text under the rectangle and I need to substract itself to go under the rectangle
                    ((text_width, text_height), o) = cv2.getTextSize(tree_type, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)

                    # Draw a title next to the green rectangle
                    cv2.putText(img_rgb, tree_type, (x, y + height + text_height), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

            # Create the final image with the rectangle and the text
            cv2.imshow("OpenCV Minecraft Tree Detection", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # The program stop if "q" is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Close the video capture and destroy the window properly
    video.release()
    cv2.destroyWindow("OpenCV Minecraft Tree Detection")


if detection_type == "image":
    print("Showing the final image...")
    detect_something_in_image(input_element, TREE_LIST)
elif detection_type == "video":
    print("Starting the real time detection... (press q to exit)")
    detect_something_in_stream(input_element, TREE_LIST)


