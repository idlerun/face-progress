#!/usr/bin/env python3

import cv2
import dlib
import numpy as np
import os, os.path, math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

MIN_SCORE = 0

colors = {
  "chin": (255, 0, 0),
  "left_eyebrow": (255, 255, 255),
  "right_eyebrow": (255, 255, 255),
  "nose_bridge": (255, 0, 0),
  "nose_tip": (255, 0, 0),
  "left_eye": (0, 255, 0),
  "right_eye": (0, 255, 0),
  "top_lip": (0, 0, 255),
  "bottom_lip": (0, 0, 255)
}

def landmarks_to_features(landmarks):
  points = [(p.x, p.y) for p in landmarks.parts()]
  return {
      "chin": points[0:17],
      "left_eyebrow": points[17:22],
      "right_eyebrow": points[22:27],
      "nose_bridge": points[27:31],
      "nose_tip": points[31:36],
      "left_eye": points[36:42],
      "right_eye": points[42:48],
      "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
      "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
  }

WINDOW_SIZE = 1024

paths = []
inDir = "input"
outDir = "output"

for file in os.listdir(inDir):
  extension = os.path.splitext(file)[1]
  if extension.lower() == ".jpg":
    paths.append(file)
paths.sort()
print("Found %d images" % len(paths))

def get_feature_pts(features, scale, name):
  points = features[name]
  if len(points) > 0:
    pts = np.array(points, np.int32)
    pts = pts * scale
    #pts = pts.reshape((-1,1,2))
    return pts
  return None

def square_up(img):
  size = int(max(img.shape[0], img.shape[1]) * 1.25)
  canvas = np.zeros((size,size,3), np.uint8) * 255
  x = int((size-img.shape[1]) / 2)
  y = int((size-img.shape[0]) / 2)
  canvas[y:y+img.shape[0],x:x+img.shape[1]] = img
  return canvas

EYE_Y = 0.25
img_trans = (0,0)
img_rot = 0
img_scale = 1.0

def guess(raw):
  global img_trans
  global img_rot
  global img_scale

  img_trans = (0,0)
  img_rot = 0
  img_scale = 1.0

  size = raw.shape[0]
  scale = 4
  img = cv2.resize(raw, (0,0), fx=1/scale, fy=1/scale)
  rgb = img[:, :, ::-1]
  dets, scores, idx = detector.run(img, 0, -1)
  match = None
  for i, d in enumerate(dets):
    if match is None or scores[i] > scores[match]:
      print("Score %f" % scores[i])
      match = i
  if match is None:
    return
  
  landmarks = predictor(rgb, dets[match])
  features = landmarks_to_features(landmarks)
  
  leftEye = get_feature_pts(features, scale, 'left_eye').mean(axis=0)
  rightEye = get_feature_pts(features, scale, 'right_eye').mean(axis=0)
  eyesCenter = ((leftEye[0] + rightEye[0]) // 2,
                (leftEye[1] + rightEye[1]) // 2)
  dY = rightEye[1] - leftEye[1]
  dX = rightEye[0] - leftEye[0]
  img_rot = np.degrees(np.arctan2(dY, dX))
  img_trans = (0.5 - eyesCenter[0] / size, EYE_Y - eyesCenter[1] / size)

def transform(img):
  size = img.shape[0]
  eyesCenter = (size/2, size * EYE_Y)
  m_trans = np.float32([[1,0,img_trans[0]*size],[0,1,img_trans[1]*size]])
  m_rot = cv2.getRotationMatrix2D(eyesCenter, img_rot, 1)
  morph = cv2.resize(img, (0,0), fx=img_scale, fy=img_scale)
  if img_scale < 1.0:
    canvas = np.zeros((size,size,3), np.uint8) * 255
    offs = int((size-morph.shape[0])/2)
    canvas[offs:offs+morph.shape[0],offs:offs+morph.shape[1]] = morph
    morph = canvas
  elif img_scale > 1.0:
    offs = int((morph.shape[0]-size) / 2)
    morph = morph[offs:offs+size,offs:offs+size]
  morph = cv2.warpAffine(morph, m_trans, (size,size), flags=cv2.INTER_CUBIC)
  morph = cv2.warpAffine(morph, m_rot, (size,size), flags=cv2.INTER_CUBIC)
  return morph

reference = cv2.resize(cv2.imread("reference.jpg"), (WINDOW_SIZE,WINDOW_SIZE))
quit = False
def process(img, path):
  global quit
  global img_trans
  global img_rot
  global img_scale

  scaled = cv2.resize(img, (WINDOW_SIZE,WINDOW_SIZE))
  show_reference = False
  while True:
    disp = transform(scaled)
    if show_reference:
      disp = cv2.addWeighted(disp,0.5,reference,0.5,0)
    dispEyes = (int(WINDOW_SIZE/2), int(WINDOW_SIZE * EYE_Y))
    s = int(WINDOW_SIZE/10)
    cv2.line(disp, (dispEyes[0]-s,dispEyes[1]), (dispEyes[0]+s,dispEyes[1]), (0,0,255), 1)
    cv2.line(disp, (dispEyes[0],dispEyes[1]), (dispEyes[0],dispEyes[1]+s), (255,0,0), 1)

    while True:
      cv2.imshow('frame', disp)
      k = cv2.waitKey(1)
      if k == -1:
        continue
      elif k == 27:
        quit = True
        return
      elif k == 32:
        return
      elif k == 13:
        morph = transform(img)
        cv2.imwrite(os.path.join(outDir, path), morph, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return

      c = chr(k)        
      if c in "wsadWSAD":
        OPS=[(0,-1),(0,1),(-1,0),(1,0), (0,-10),(0,10),(-10,0),(10,0)]
        x = "wsadWSAD".find(c)
        img_trans = (img_trans[0] + OPS[x][0] / WINDOW_SIZE, img_trans[1] + OPS[x][1]  / WINDOW_SIZE)
        break
      elif c in "qeQE":
        OPS=[-1,1,-5,5]
        x = "qeQE".find(c)
        img_rot = img_rot+OPS[x]
        break
      elif c in "tgTG":
        OPS=[-0.01,0.01,-0.05,0.05]
        x = "tgTG".find(c)
        img_scale = img_scale+OPS[x]
        break
      elif c == 'r':
        show_reference = not show_reference
        break
      else:
        print("k=%d" % k)

for path in paths:
  print("Reading %s" % path)
  img = cv2.imread(os.path.join(inDir, path))
  if img is None:
    print("Failed to read %s" % path)
    continue
  img = square_up(img)
  size = img.shape[0]
  eyesCenter = (size/2, size * EYE_Y)
  guess(img)
  process(img, path)
  if quit:
    break

cv2.destroyAllWindows()
