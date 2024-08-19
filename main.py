""" Example for implementing AOI@home """

from AOI.pcb_aoi import pcb
from AOI.compare.compare import compare

import cv2

pattern = [
    [0, 1, 2],
    [3, 4, 5]
]

if __name__ == "__main__" :

    # Generate the golden board
    pcb0 = pcb("pi_pcb0", pattern)
    #pcb1.pcb.show(1)
    #pcb1.pcb.mask.show(1)

    comparison = compare(pcb0)

    # Generate the board to compare to the golden
    pcb1 = pcb("pi_pcb1", pattern)

    comparison.compare_to_golden(pcb1)
    comparison.comparison.show(1)


    cv2.waitKey(0)