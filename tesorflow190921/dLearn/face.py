import cv2

class FaceModel:
    def __init__(self):
        self._fname = './data/family.jpg'

    def original(self):
        img = cv2.imread(self._fname)
        cv2.imshow('family', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()