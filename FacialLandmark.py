import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def createBox(img, points, scale = 5):
  bBox = cv2.boundingRect(points)
  x, y, w, h = bBox
  imgCrop = img[y-5:y+h+5, x-5:x+w+5]
  imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
  return imgCrop


img = cv2.imread('Bae Suzy.jpg')
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
imgOriginal = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgGray)

for face in faces:
  x1, y1 = face.left(), face.top()
  x2, y2 = face.right(), face.bottom()
  imgOriginal = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
  landmarks = predictor(imgGray, face)
  myPoints = []
  for n in range(68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    myPoints.append([x, y])
    # cv2.circle(imgOriginal, (x, y), 2, (50, 50, 255), cv2.FILLED)
    # cv2.putText(imgOriginal, str(n), (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)
  
  # convert to numpy array
  myPoints = np.array(myPoints)
  imgLeftEye = createBox(img, myPoints[36:42]) # right-side array is excluded
  imgLips = createBox(img, myPoints[48:61], 3)
  cv2.imshow('Left Eye', imgLeftEye)
  cv2.imshow('Lips', imgLips)
  print(myPoints)

cv2.imshow('Original', imgOriginal)
cv2.waitKey(0)