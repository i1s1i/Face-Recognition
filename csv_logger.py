import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, filename="detection_log.csv"):
        self.filename = filename
        self.file_exists = os.path.isfile(self.filename)
        
        if not self.file_exists:
            # Create file and write headers
            with open(self.filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Num_Faces", "Emotions", "Num_Hands", "Total_Fingers", "Hand_Gestures"])
                
    def log_data(self, faces_data, hands_data):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        num_faces = len(faces_data)
        if num_faces > 0:
            emotions = ", ".join([f["emotion"] for f in faces_data])
        else:
            emotions = "None"
            
        num_hands = hands_data.get("num_hands", 0)
        total_fingers = hands_data.get("total_fingers", 0)
        
        gestures_list = hands_data.get("gestures", [])
        if gestures_list:
            gestures = ", ".join(gestures_list)
        else:
            gestures = "None"
            
        with open(self.filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, num_faces, emotions, num_hands, total_fingers, gestures])
