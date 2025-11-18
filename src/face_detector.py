import cv2

class FaceDetector:
    def __init__(self, cascade_path: str):
        """Haar Cascade 얼굴 검출기 초기화"""
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """프레임에서 얼굴을 검출하여 (x, y, w, h) 리스트 반환"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        return faces
