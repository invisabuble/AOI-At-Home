from .methods import methods_parent
import cv2
import numpy as np

class image (methods_parent) :
    """ Class for managing images """

    def __init__ (self, name, image):
        super().__init__(name, image)

        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.component_mask = self.create_component_mask()

    
    def create_component_mask (self) :
        """ Create a component mask from the class image """

        Filters = [[(40,79,0),(179,255,255)],[(27,82,14),(84,255,191)],[(36,61,35),(91,255,156)],[(22,77,0),(87,255,186)]]

        new_mask = np.zeros((self.height, self.width), np.uint8)

        for Filter in Filters :
            # Iterate through the filters and add the hsv values in range to the mask
            in_range = cv2.inRange(self.hsv_image, Filter[0], Filter[1])
            new_mask = cv2.addWeighted(new_mask, 1, in_range, 1, 0)

        # Invert the new mask
        new_mask = (255-new_mask)

        cv2.imshow(f"{self.name}-mask", new_mask)

        return new_mask

