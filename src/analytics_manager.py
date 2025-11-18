from datetime import datetime
import pandas as pd
from collections import deque

class AnalyticsManager:
    def __init__(self, max_history=300):
        # 최근 감정 기록 (최대 300개 → 약 10초 분량)
        self.history = deque(maxlen=max_history)

        # 감정별 누적 카운트
        self.emotion_counts = {
            "neutral": 0,
            "happy": 0,
            "sad": 0,
            "surprise": 0,
            "fear": 0,
            "disgust": 0,
            "angry": 0,
            "contempt": 0
        }

    def update(self, label, confidence):
        """감정 기록 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.history.append({
            "timestamp": timestamp,
            "emotion": label,
            "confidence": confidence
        })

        # 카운트 업데이트
        if label in self.emotion_counts:
            self.emotion_counts[label] += 1

    def get_recent_df(self):
        """최근 감정 기록 Pandas DataFrame 반환"""
        if not self.history:
            return pd.DataFrame(columns=["timestamp", "emotion", "confidence"])
        return pd.DataFrame(list(self.history))

    def save_to_csv(self, path="logs/emotions.csv"):
        """로그 전체를 CSV로 저장"""
        df = self.get_recent_df()
        df.to_csv(path, index=False)
