from .methods import methods_parent
import cv2

class mask (methods_parent) :
    """ Class for managing masks """

    def __init__ (self, name, image):
        super().__init__(name, image)