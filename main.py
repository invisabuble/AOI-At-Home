""" Example for implementing AOI@home """

from AOI.pcb_aoi import pcb

from AOI.image.image_aoi import image
from AOI.image.mask_aoi import mask

import cv2

pattern = [
    [0, 1, 2],
    [3, 4, 5]
]

if __name__ == "__main__" :

    pcb0 = pcb("pcb0", pattern)

    pcb0.pcb.show(1)

    cv2.waitKey(0)