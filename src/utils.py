import cv2

def open_camera(index=0, width=640, height=480):
    """웹캠 열기"""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def resize_frame(frame, width=640):
    """프레임 너비 기준 리사이즈 (FPS 개선용)"""
    h, w = frame.shape[:2]
    ratio = width / w
    new_size = (width, int(h * ratio))
    return cv2.resize(frame, new_size)

def crop_face(frame, x, y, w, h):
    """얼굴 ROI 추출"""
    return frame[y:y+h, x:x+w]
