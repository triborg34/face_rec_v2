import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from collections import deque
from torchvision import transforms
import torch
from face_alignment.alignment import norm_crop
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
import logging
import warnings
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RTSPVideoStream:
    def __init__(self, rtsp_url, buffer_size=30):
        self.rtsp_url = rtsp_url
        self.frame_buffer = deque(maxlen=buffer_size)
        self.stopped = False
        
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.last_frame_time = 0
        self.target_fps = 20
        self.frame_interval = 1.0 / self.target_fps

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading from stream. Reconnecting...")
                self._reconnect()
                continue
            
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                self.frame_buffer.append(frame)
                self.last_frame_time = current_time

    def _reconnect(self):
        self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def read(self):
        if self.frame_buffer:
            return True, self.frame_buffer[-1]
        return False, None

    def stop(self):
        self.stopped = True
        self.cap.release()


class FaceDetectionSystem:
    def __init__(self, model_path, feature_path, arcface_model_path):
        """Initialize the face detection and recognition system."""
        # YOLO face detection model
        self.model = YOLO(model_path)
        self.model.conf = 0.3
        self.model.iou = 0.5
        
        # ArcFace model for recognition
        self.recognizer = iresnet_inference(
            model_name="r100", path=arcface_model_path, device=device
        )

        # Load precomputed face features
        self.images_names, self.images_embs = read_features(feature_path)

        # Track frame processing times
        self.frame_times = deque(maxlen=30)

    def calculate_fps(self):
        """Calculate current FPS."""
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                return len(self.frame_times) / time_diff
        return 0

    @torch.no_grad()
    def get_feature(self, face_image):
        """Extract features from a face image."""
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_tensor = preprocess(face_image).unsqueeze(0).to(device)
        emb_face = self.recognizer(face_tensor).cpu().numpy()
        return emb_face / np.linalg.norm(emb_face)

    def recognize_face(self, face_image):
        """Recognize the given face."""
        query_emb = self.get_feature(face_image)
        score, id_min = compare_encodings(query_emb, self.images_embs)
        name = self.images_names[id_min] if score[0] >= 0.25 else "Unknown"
        return name, score[0]

    def process_frame(self, frame):
        """Detect faces and recognize them."""
        self.frame_times.append(time.time())
        fps = self.calculate_fps()

        # Detect faces
        results = self.model(frame)

        for result in results:
            for box in result.boxes:
                # Extract bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])

                if confidence < 0.3:
                    continue

                # Crop the face for recognition
                face_image = frame[y1:y2, x1:x2]
                if face_image.size == 0:
                    continue

                # Recognize the face
                name, score = self.recognize_face(face_image)

                # Draw bounding box and label
                label = f"{name} ({score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"name : {name}")

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame


def main():
    # Paths
    video_source = "rtsp://192.168.1.7:554/stream"
    # video_source = 0
    model_path = "yolov8n-face.pt"
    feature_path = "./datasets/face_features/feature"
    arcface_model_path = "./face_recognition/arcface/weights/arcface_r100.pth"

    # Initialize system
    face_system = FaceDetectionSystem(model_path, feature_path, arcface_model_path)
    stream = RTSPVideoStream(video_source).start()

    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                
                continue

            # Process frame
            processed_frame = face_system.process_frame(frame)
            cv2.imshow("Face Detection and Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
