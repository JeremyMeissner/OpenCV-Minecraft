# OpenCV-Minecraft Tutorial

## Capture a video
 First you need to capture an video on Minecraft

 > I've personnaly capture a 2 minutes long video of me flying and walking over a world in minecraft without any trees


## Convert to multiple images
After that, you need to convert the mp4 video into multiple image with `ffmpeg` (you can install it using `apt-get install ffmpeg`)

```bash
ffmpeg -i input.mp4 -qscale:v 2 -vf fps=3 ./negativeImage/out%d.jpg
```
* `-vf fps=3` is to choose how much image do you want every frame, this one output 3 images every seconds
* `-qscale:v 2` is to choose the quality (2 is excellent quality and 31 is the worst quality)


You need to put this images into the `./negativeImage` folder.

Now go the the root of the folder and use this command to create a file that contain all the references of the images
```bash
cd negativeImage
ls *.jpg > negatives.txt
```

You now need to install the `libopencv-dev` library using `apt-get install libopencv-dev` to continue


```bash
opencv_createsamples -img trees/1.jpg -bg negativeImage/negatives.txt -info sampleImageTest/cropped1.txt -num 128 -maxxangle 0.0 -maxyangle 0.0 -maxzangle 0.3 -bgcolor 255 -bgthresh 8 -w 48 -h 48
```

> I've edit the `negativeImage/negatives.txt` and remove the relative folder on each line


Next, you need to collect all the description files and combine into one file
```bash
cat sampleImageTest/cropped*.txt > sampleImageTest/positives.txt
```
Then combine all the images into a vec file
```bash
opencv_createsamples -info sampleImageTest/positives.txt -bg negativeImageDirectory/negatives.txt -vec cropped.vec -num 250 -w 48 -h 48
```
* -num 250 is the number of positives images you have

And finally train our Haar classifier with the following command:
```
cd negativeImage 
opencv_traincascade -data ../classifier -vec ../cropped.vec -bg negatives.txt -numPos 200 -numNeg 600 -numStages 10 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -featureType HAAR -minHitRate 0.995 -maxFalseAlarmRate 0.5 -w 48 -h 48
```
* -numPos 200 is the number of positives images you have (with a margin because opencv can take more positive images that you have some times)
* -numNeg 600 is the number of negatives images
