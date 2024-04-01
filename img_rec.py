import cv2
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import threading
import json

class ObjectDetector:
    def __init__(self):
        # Initialize the model and processor here because threading doesn't have the same issues as multiprocessing
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.latest_results = ""
        self.running = False

    def detect_objects(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        detected_objects = [self.model.config.id2label[label.item()] for label in results["labels"]]
        return detected_objects

    def _run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            detected_objects = self.detect_objects(frame)
            self.latest_results = json.dumps(detected_objects)
            # Visualization or other processing code can go here...

        cap.release()
        cv2.destroyAllWindows()

    def stream_and_detect(self,input_text=None):
        self.running = True
        detection_thread = threading.Thread(target=self._run_detection)
        detection_thread.start()

    def stop_object_detection(self, input_text=None):
        self.running = False

    def get_latest_results(self, input_text=None):
        if self.latest_results:
            results = json.loads(self.latest_results)
            print("Getting latest results...", results)
            return results
        else:
            return "No objects detected or object detection not running."
