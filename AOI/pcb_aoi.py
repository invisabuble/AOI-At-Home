from AOI.image.image_aoi import image
from AOI.image.mask_aoi import mask
import cv2

class pcb :
    """ Creates a pcb class """

    def __init__ (self, name) :
        """ Initialise the pcb class """

        self.pcb_images = f"pcbs/{name}"