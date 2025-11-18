import cv2
import numpy as np

class EmotionClassifier:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # FERPlus 모델의 원래 라벨(영어)
        self.labels = [
            "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"
        ]

    def predict(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return "unknown", 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        blob = cv2.dnn.blobFromImage(gray, 1.0/255, (64, 64))
        self.net.setInput(blob)
        output = self.net.forward()

        idx = np.argmax(output)
        label = self.labels[idx]
        confidence = float(output[0][idx])

        return label, confidence
