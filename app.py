import cv2
import time
from hand_detector import HandDetector
from face_detector import FaceDetector
from csv_logger import CSVLogger

class DetectionPipeline:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.face_detector = FaceDetector()
        self.logger = CSVLogger()
        self.cap = cv2.VideoCapture(0)
        self.last_log_time = time.time()
        
    def run(self):
        print("Starting camera feed. Press ESC to exit.")
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
                
            # Flip image for a selfie-view display
            img = cv2.flip(img, 1)
            
            # 1. Process Hands
            img, hands_data = self.hand_detector.analyze_hands(img)
            
            # 2. Process Faces
            img, faces_data = self.face_detector.analyze_faces(img)
            
            # 3. Print info on screen
            hands_str = f"Hands: {hands_data['num_hands']} | Fingers: {hands_data['total_fingers']}"
            cv2.putText(img, hands_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            gestures = hands_data.get('gestures', [])
            gestures_str = f"Gestures: {', '.join(gestures) if gestures else 'None'}"
            cv2.putText(img, gestures_str, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            cv2.imshow("Multi-Detection System", img)
            
            # 4. Log data every 1 second
            current_time = time.time()
            if current_time - self.last_log_time > 1.0:
                self.logger.log_data(faces_data, hands_data)
                self.last_log_time = current_time
            
            # Press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = DetectionPipeline()
    pipeline.run()
