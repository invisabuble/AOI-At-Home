from AOI.image.image_aoi import image
from AOI.image.mask_aoi import mask
import cv2
import numpy as np

class pcb :
    """ Creates a pcb class """

    def __init__ (self, name, pattern = 0) :
        """ Initialise the pcb class """

        self.im_type = ".png"
        self.im_path = f"pcbs/{name}/"
        self.name = name
        self.pattern = pattern

        self.pcb_image = self.assemble_board_image()
        self.pcb = image(name, self.pcb_image)


    def image_stitcher (self, im1, im2) :
        """ Stitch two images together """

        # Create a SIFT detector object
        sift = cv2.SIFT_create()

        # Convert images to grayscale and detect keypoints and descriptors
        im1_kp, im1_desc = sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), None)
        im2_kp, im2_desc = sift.detectAndCompute(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), None)

        # Create a brute-force matcher object
        matcher = cv2.BFMatcher()

        # Perform k-nearest neighbor matching to find matches between descriptors
        matches = matcher.knnMatch(im1_desc, im2_desc, k = 2)

        # Apply ratio test to filter out false matches
        good_matches = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        # Check if there are enough good matches to compute homography
        if len(good_matches) < 4:
            print("Not enough good matches found to form a homography matrix.")
            exit(0)

        # Extract points from the good matches
        im1_p = []
        im2_p = []

        for match in good_matches:
            im1_p.append(im1_kp[match[0].queryIdx].pt)
            im2_p.append(im2_kp[match[0].trainIdx].pt)

        # Convert points to float32
        im1_p = np.float32(im1_p)
        im2_p = np.float32(im2_p)

        # Compute the homography matrix using RANSAC
        (homography, status) = cv2.findHomography(im2_p, im1_p, cv2.RANSAC, 4.0)

        # Get the dimensions of the second image
        (h, w) = im2.shape[:2]

        # Define the corners of the second image
        initial_matrix = np.array([[0, w - 1, w - 1, 0],
                                [0, 0, h - 1, h - 1],
                                [1, 1, 1, 1]])

        # Apply the homography matrix to the corners to get the transformed corners
        final_matrix = np.dot(homography, initial_matrix)

        # Normalize the coordinates
        [x, y, c] = final_matrix
        x = np.divide(x, c)
        y = np.divide(y, c)

        # Find the bounding box of the transformed image
        minX, maxX = int(round(min(x))), int(round(max(x)))
        minY, maxY = int(round(min(y))), int(round(max(y)))

        # Calculate the new width and height of the stitched image
        new_width = maxX
        new_height = maxY
        cor = [0,0]

        # Adjust width and height if the bounding box is outside the original image
        if minX < 0:
            new_width -= minX
            cor[0] = abs(minX)

        if minY < 0:
            new_height -= minY
            cor[1] = abs(minY)

        # Ensure the stitched image is large enough to fit both images
        if new_width < im1.shape[1] + cor[0]:
            new_width = im1.shape[1] + cor[0]

        if new_height < im1.shape[0] + cor[1]:
            new_height = im1.shape[0] + cor[1]

        # Adjust coordinates by the offset
        x = np.add(x, cor[0])
        y = np.add(y, cor[1])

        # Define the corners of the first image
        im1_corners = np.float32([[0, 0],
                            [w - 1, 0],
                            [w - 1, h - 1],
                            [0, h - 1]])

        # Compute the perspective transform matrix
        final_points = np.float32(np.array([x, y]).transpose())
        homography_matrix = cv2.getPerspectiveTransform(im1_corners, final_points)

        # Warp the second image to align with the first image
        final_image = cv2.warpPerspective(im2, homography_matrix, (new_width,new_height))

        # Place the first image on top of the warped second image
        final_image[cor[1]:cor[1]+im1.shape[0], cor[0]:cor[0]+im1.shape[1]] = im1

        # Return the stitched image
        return final_image


    def crop_black_edges (self, image) :
        """ Crop black artifacts off the edge of an image """

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a mask where non black pixels are white and black pixels are black
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Iterate through the image and find the rightmost black pixel on the left
        # and the leftmost black pixel on the right

        rightmost = 0
        leftmost = 0

        # Iterate through each row in the image
        for row in mask:
            row = np.asarray(row)

            # If theres 255 in the row then find the rightmost and leftmost
            if 255 in row:
                
                reversed_row = row[::-1]

                right = np.argmax(row == 255)
                rightmost = max(right, rightmost)

                left = np.argmax(reversed_row == 255)
                leftmost = max(left, leftmost)

        width = image.shape[1]
        height = image.shape[0]
        # Crop the black edges off of the image
        image = image[0:height, rightmost:width-leftmost]

        return image


    def assemble_board_image (self) :
        """ Assembles a full board image from the pattern """

        # If theres a pattern defined then assemble the full board image
        if self.pattern:

            row_images = []

            # Iterate through each row in the pattern
            for row in self.pattern :

                row_image = cv2.imread(f"{self.im_path}{row[0]}{self.im_type}")

                # For each column in each row, stitch the images together
                for column in row[1:]:

                    column_image = cv2.imread(f"{self.im_path}{column}{self.im_type}")
                    row_image = self.image_stitcher(row_image, column_image)

                # Rotate 90 degrees counter-clockwise for later row assembly
                row_image = cv2.rotate(row_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # Crop the black edges off of the row
                row_image = self.crop_black_edges(row_image)
                row_images.append(row_image)

            # Stitch every row in row_images together 
            final_image = row_images[0]

            for row in row_images[1:] :

                final_image = self.image_stitcher(final_image, row)

            # Rotate the final image back to its initial rotation
            final_image = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)
            final_image = self.crop_black_edges(final_image)

        else:

            final_image = cv2.imread(f"{self.im_path}0{self.im_type}")

        return final_image
        

        