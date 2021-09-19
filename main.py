import torch
import numpy as np
import cv2
import PIL.Image as Image
import os
from detector import FaceDetector

if __name__ == '__main__':
    fileList = os.listdir('./test')
    for file in fileList:
        img = cv2.imread('./test/' + file)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = FaceDetector()
        img = t.detectMask(img)
        Image._show(img)
