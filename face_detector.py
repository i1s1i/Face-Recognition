import cv2
from deepface import DeepFace
import threading

class FaceDetector:
    def __init__(self):
        # We use standard haar cascades for faster face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = []
        self.lock = threading.Lock()
        self.is_analyzing = False
        
    def analyze_emotions_thread(self, frame):
        try:
            # We use enforce_detection=False to avoid crashing if face is not perfectly clear
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if not isinstance(results, list):
                results = [results]
                
            mapped_emotions = []
            for face in results:
                dom_emotion = face.get('dominant_emotion', 'neutral')
                
                # Map to required requirements
                if dom_emotion == 'happy':
                    mapped = "happy (you are beautiful)"
                elif dom_emotion == 'sad':
                    mapped = "sad/cry"
                elif dom_emotion == 'angry':
                    mapped = "angry"
                elif dom_emotion in ['fear', 'disgust', 'surprise', 'neutral']:
                    mapped = "normal"
                else:
                    mapped = dom_emotion
                
                region = face.get('region', {})
                mapped_emotions.append({"emotion": mapped, "region": region})
                
            with self.lock:
                self.emotions = mapped_emotions
        except Exception as e:
            pass
        finally:
            self.is_analyzing = False

    def analyze_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # We start an async thread so video feed doesn't freeze during heavy emotion inference
        if not self.is_analyzing and len(faces) > 0:
            self.is_analyzing = True
            img_copy = img.copy()
            threading.Thread(target=self.analyze_emotions_thread, args=(img_copy,), daemon=True).start()
            
        with self.lock:
            current_emotions = self.emotions.copy()
            
        face_data = []
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            matched_emotion = "analyzing..."
            max_iou = 0
            
            # Match the emotion region to the haar cascade bounding box using IoU
            for emo in current_emotions:
                rx = emo['region'].get('x', 0)
                ry = emo['region'].get('y', 0)
                rw = emo['region'].get('w', 0)
                rh = emo['region'].get('h', 0)
                
                x_left = max(x, rx)
                y_top = max(y, ry)
                x_right = min(x+w, rx+rw)
                y_bottom = min(y+h, ry+rh)
                
                if x_right > x_left and y_bottom > y_top:
                    inter_area = (x_right - x_left) * (y_bottom - y_top)
                    iou = inter_area / float(w*h + rw*rh - inter_area)
                    if iou > max_iou and iou > 0.1:
                        max_iou = iou
                        matched_emotion = emo['emotion']
            
            face_data.append({"bbox": (x,y,w,h), "emotion": matched_emotion})
            cv2.putText(img, matched_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return img, face_data
