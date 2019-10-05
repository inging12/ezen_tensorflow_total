import cv2

class FaceDetect:
    def __init__(self):
        self._cascade = './data/haarcascade_frontalface_alt.xml'
        self._girl = './data/girl.jpg'

    def read_file(self):
        cascade = cv2.CascadeClassifier(self._cascade)
        img = cv2.imread(self._girl)
        face = cascade.detectMultiScale(img, minSize=(150,150))
        if len(face) == 0:
            print('얼굴 인식 실패')
            quit()
        for(x,y,w,h) in face:
            print('얼굴의 좌표 = ', x, y, w, h)
            red = (0, 0, 255)
            cv2.rectangle(img, (x,y), (x+w, y+h), red, thickness=20)

        cv2.imwrite('girl-face.png', img)
        cv2.imshow('girl-face', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
