import streamlit as st
import cv2
import numpy as np

from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier
from analytics_manager import AnalyticsManager
from utils import open_camera, resize_frame, crop_face

st.set_page_config(page_title="Real-time Emotion Analytics", layout="wide")
st.title("😊 Real-time Face Emotion Analytics Dashboard")

st.write("모듈 초기화 중...")


face_detector = FaceDetector("resources/haarcascade_frontalface_default.xml")
emotion_classifier = EmotionClassifier("models/emotion-ferplus.onnx")
analytics = AnalyticsManager()

st.success("모듈 로드 완료!")



st.subheader("📷 테스트 이미지 감정 분석")

uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 얼굴 검출
    faces = face_detector.detect(img)

    if len(faces) == 0:
        st.warning("얼굴을 찾을 수 없습니다.")
    else:
        for (x, y, w, h) in faces:
            roi = crop_face(img, x, y, w, h)
            label, conf = emotion_classifier.predict(roi)
            analytics.update(label, conf)

            # 화면에 그리기용
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({conf:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="분석 결과")



st.subheader("📊 최근 감정 기록 (DEBUG 출력)")

df = analytics.get_recent_df()
st.dataframe(df)
