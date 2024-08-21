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
            self.image = cv2.erode(self.image, kernel, iterations=1)

        elif morph == "dilate":
            self.image = cv2.dilate(self.image, kernel, iterations=1)

        else:
            raise ValueError(f"Unknown morphology type : {morph}")
        

    def get_mask_objects (self) :
        """ Find all the objects on a mask """

        # Find contours in the image
        contours, hierarchy = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_coordinates = []

        for contour in contours:
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            object_coordinate = [
                (x,y),
                (x+w,y+h)
                ]
            
            object_coordinates.append(object_coordinate)

        return object_coordinates


        