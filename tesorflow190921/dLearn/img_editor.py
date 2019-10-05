import cv2

class ImgEditor:
    def __init__(self):
        self._fname = './data/test.jpg'

    def original(self):
        img = cv2.imread(self._fname)
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def negative(self):
        img = cv2.imread(self._fname)
        img = 255 - img # 네거티브 반전
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bgr2gray(self):
        img = cv2.imread(self._fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 역치 지정하기
        th = 90
        img[img > th] = 255
        img[img < th] = 0
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def img_cut(self):
        img = cv2.imread(self._fname)
        im2 = img[150:450, 150:450]
        im2 = cv2.resize(im2, (400,400))
        # 크기 변경한 이미지 저장
        cv2.imwrite('img_small.png', im2)
        cv2.imshow('test', im2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()