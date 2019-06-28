---
page: https://idle.run/face-progress
title: Face Alignment for Growth Progress
tags: python opencv
date: 2019-05-20
---

Uses OpenCV to align the face in a series of photos with optional manual fine tuning.

## Prerequisites

### DLib Feature Model

Download the 68 feature model for face shape prediction (into the repository alongside `progress.py`)

```
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```


## Environment Setup

* Create a directory `input` filled with images to be processed
* Create a directory `output` for the result images to be written
* Choose a single reference photo to use and put it in the `input` directory
* Run `./progress.py`
* Adjust the image as desired using *Controls* described below and save
* Copy the resulting image from `output` and overwrite `reference.jpg` with the new image


## Usage

* Load up the `input` directory with all images and run `./progress.py` to process them
* Use the *Controls* described below to align the images as closely as possible
* Using [fix-photo-dates](https://idle.run/fix-photo-dates) on the output may be helpful


## Controls

* `q` and `e` rotate left and right
* `wsad` pan image
* `t` and `g` zoom in and out (Ideally all images should be taken from the same distance so that zoom is not required)
* `r` overlay reference image with working image for alignment assist
* `<enter>` save image and open next
* `<spacebar>` skip image and open next
  

## Syncing from Android

Here is a simple script to sync new photos only from an Android device over `adb`

In this example I'm using the `Open Camera` app set to save images to the folder `/sdcard/Progress` - adjust as required.

```
cd input
adb ls /sdcard/Progress | grep -e "\.jpg$" | cut -d" " -f4 | grep -v -f <(echo __; ls -1; ls -1 ../done) | xargs -I{} -n 1 adb pull /sdcard/Progress/{}
```

## Convert to Video

Here is an example of converting the images to a video. Adjust the cropping params as required.

```
ffmpeg -r 10 -f image2 -pattern_type glob \
  -i 'output/*.jpg' -filter:v 'crop=in_w*6/12:in_h*7/12:in_w*3/12:in_h*1/12' \
  -an -c:v libx264 -pix_fmt yuv420p -crf 25 \
  vid.mp4
```
