import cv2

class methods_parent (object) :
    """ Parent class containing methods for manipulating images """

    def __init__ (self, name, image) :
        """ Initialise the methods class """

        self.image = image
        self.name = name
        self.width = image.shape[1]
        self.height = image.shape[0]


    def scale (self, scale_factor) :
        """ Scale the image within the object, keeping the aspect ratio """

        new_width = int(self.width * scale_factor)
        new_height = int(self.height * scale_factor)

        self.resize(new_width, new_height)


    def resize (self, width, height) :
        """ Resize the image within the object """

        self.width = width
        self.height = height

        self.image = cv2.resize(self.image, (width, height))


    def show (self, scale_factor = 1) :
        """ Show the image within the methods object, scaling if required """

        if scale_factor != 1 :
            self.scale(scale_factor)

        cv2.imshow(self.name, self.image)

        

        
