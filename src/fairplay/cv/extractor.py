import cv2
import numpy as np
import matplotlib.pyplot as plt
from fairplay.cv.lines import find_xaxis, find_yaxis

class Extractor:

    def __init__(self, cv2_img):
        self.img = cv2_img
        (self.m, self.n, self.p) = self.img.shape
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.bw = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
        self.imlabel = self.img.copy()


    def show(self, imtype="imlabel"):
        if imtype == "imlabel":
            fig = plt.figure()
            implot = plt.imshow(self.imlabel)
            return fig, implot
        elif imtype == "bw":
            fig = plt.figure()
            implot = plt.imshow(self.bw, cmap="gray")
            return fig, implot
        else:
            return
        

    def find_axes(self):
        self.xax = find_xaxis(self.bw)
        self.yax = find_yaxis(self.bw)
        self.imlabel = cv2.line(
            self.imlabel,
            (self.xax[0], self.xax[1]),
            (self.xax[2], self.xax[3]),
            (255, 0, 0),
            4
            )
        self.imlabel = cv2.line(
            self.imlabel,
            (self.yax[0], self.yax[1]),
            (self.yax[2], self.yax[3]),
            (255, 0, 0),
            4
            )
        self.roi = np.array([
            min(self.yax[0], self.yax[2]),  # furthest left point of y axis
            min(self.yax[1], self.yax[3]),  # highest point on y axis
            max(self.xax[0], self.xax[2]),  # furthest right point of x axis
            max(self.xax[1], self.xax[3])   # lowest point on x axis
        ])
        return self
    