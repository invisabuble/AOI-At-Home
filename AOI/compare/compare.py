from ..image.mask_aoi import mask

import cv2
import numpy as np

class compare (object) :
    """ Compare an input mask to the golden """

    def __init__ (self, golden_object) :
        """ Initialise the comparison object """

        self.golden_object = golden_object
        self.golden_width = golden_object.pcb.mask.width
        self.golden_height = golden_object.pcb.mask.height

        self.comparison = None

    def compare_to_golden (self, test_board) :
        """ Compare a test board to the golden """

        # If the test board mask is a different size to the golden, resize it
        test_width = test_board.pcb.mask.width
        test_height = test_board.pcb.mask.height

        if test_width != self.golden_width or test_height != self.golden_height :
            test_board.pcb.mask.resize(self.golden_width, self.golden_height)

        # Get the difference between the masks
        abs_diff = cv2.absdiff(
            self.golden_object.pcb.mask.image,
            test_board.pcb.mask.image
            )
        
        # Create mask object from absolute difference
        self.comparison = mask(f"{self.golden_object.name}->{test_board.name}-comparison", abs_diff)

        # Processing for stray pixels, and strengthening detections
        self.comparison.morphology("erode", strength=5)
        self.comparison.morphology("dilate", strength=5)

        # Get contours from the differences, then coordinates from the contours
        
