import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        # We use MediaPipe Tasks API for Python 3.12+ compatibility
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.results = None
        
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        self.results = self.landmarker.detect(mp_image)
        
        if self.results.hand_landmarks and draw:
            h, w, c = img.shape
            for hand_landmarks in self.results.hand_landmarks:
                # Draw landmarks manually to ensure cross-version compatibility
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), cv2.FILLED)
                
                for connection in self.HAND_CONNECTIONS:
                    lm1 = hand_landmarks[connection[0]]
                    lm2 = hand_landmarks[connection[1]]
                    cx1, cy1 = int(lm1.x * w), int(lm1.y * h)
                    cx2, cy2 = int(lm2.x * w), int(lm2.y * h)
                    cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                    
        return img

    def get_fingers_state(self, hand_landmarks):
        # returns [thumb, index, middle, ring, pinky]
        index_extended = hand_landmarks[8].y < hand_landmarks[6].y
        middle_extended = hand_landmarks[12].y < hand_landmarks[10].y
        ring_extended = hand_landmarks[16].y < hand_landmarks[14].y
        pinky_extended = hand_landmarks[20].y < hand_landmarks[18].y
        
        pinky_base_x = hand_landmarks[17].x
        thumb_tip_x = hand_landmarks[4].x
        thumb_base_x = hand_landmarks[2].x
        
        thumb_dist = abs(thumb_tip_x - pinky_base_x)
        base_dist = abs(thumb_base_x - pinky_base_x)
        thumb_extended = thumb_dist > base_dist
        
        return [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

    def detect_gesture(self, hand_landmarks):
        states = self.get_fingers_state(hand_landmarks)
        thumb, index, middle, ring, pinky = states
        
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        dist_ok = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        
        if dist_ok < 0.05 and middle and ring and pinky:
            return "OK"
            
        if thumb and not index and not middle and not ring and not pinky:
            if thumb_tip.y > hand_landmarks[2].y:
                return "Not OK"
            else:
                return "Thumb Up"
                
        if not thumb and index and middle and not ring and not pinky:
            return "Peace"
            
        if thumb and index and not middle and not ring and pinky:
            return "I Love You"
            
        if thumb and index and not middle and not ring and not pinky:
            return "Gun Hand"
            
        if thumb and index and middle and ring and pinky:
            return "Freedom"
            
        return "Unknown"

    def analyze_hands(self, img):
        img = self.find_hands(img)
        analysis = {
            "num_hands": 0,
            "total_fingers": 0,
            "gestures": [],
            "is_heart": False
        }
        
        if self.results and self.results.hand_landmarks:
            all_landmarks = self.results.hand_landmarks
            analysis["num_hands"] = len(all_landmarks)
            
            # Check for heart
            if len(all_landmarks) == 2:
                hand1 = all_landmarks[0]
                hand2 = all_landmarks[1]
                
                t_dist = math.hypot(hand1[4].x - hand2[4].x, hand1[4].y - hand2[4].y)
                i_dist = math.hypot(hand1[8].x - hand2[8].x, hand1[8].y - hand2[8].y)
                
                if t_dist < 0.1 and i_dist < 0.1:
                    analysis["is_heart"] = True
                    analysis["gestures"].append("Heart")
            
            for hand_landmarks in all_landmarks:
                states = self.get_fingers_state(hand_landmarks)
                analysis["total_fingers"] += sum(states)
                
                if not analysis["is_heart"]:
                    gesture = self.detect_gesture(hand_landmarks)
                    analysis["gestures"].append(gesture)
                    
        return img, analysis
