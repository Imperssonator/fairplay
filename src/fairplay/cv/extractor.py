import cv2
from fairplay.cv.lines import find_xaxis

class Extractor:
    def __init__(self, cv2_img):
        self.img = cv2_img
        (self.m, self.n, self.p) = self.img.shape
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.bw = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)

    def find_xaxis(self):
        self.xax_pts = find_xaxis(self.bw)
        return self