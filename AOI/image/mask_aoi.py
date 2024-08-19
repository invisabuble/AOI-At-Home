from .methods import methods_parent
import cv2
import numpy as np

class mask (methods_parent) :
    """ Class for managing masks """

    def __init__ (self, name, image):
        super().__init__(f"{name}-mask", image)


    def morphology (self, morph, strength = 3, iterations = 1) :
        """ Erode or dilate the mask """
        kernel = np.ones((strength, strength), np.uint8)
        if morph == "erode" :
            self.erode(kernel, iterations)
        elif morph == "dilate":
            self.dilate(kernel, iterations)


    def erode (self, kernel, iterations) :
        """ Erode the mask """
        self.image = cv2.erode(self.image, kernel, iterations=1)


    def dilate (self, kernel, iterations) :
        """ Dilate the mask """
        self.image = cv2.dilate(self.image, kernel, iterations=1)