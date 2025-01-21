import cv2
import numpy as np
from collections import OrderedDict
import time
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            for col in unused_cols:
                self.register(input_centroids[col])
        return self.objects

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
cap = cv2.VideoCapture('crowd1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
up_count = 0
down_count = 0
ct = CentroidTracker()
previous_y = {}
previous_centroids = {}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    centroids = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    objects = ct.update(centroids)
    current_time = time.time()

    total_speed = 0
    count_people = 0

    for (object_id, centroid) in objects.items():
        current_y = centroid[1]

        if object_id in previous_y:
            prev_y = previous_y[object_id]
            if prev_y > line_position and current_y < line_position:
                up_count += 1
            elif prev_y < line_position and current_y > line_position:
                down_count += 1

            if (prev_y <= line_position and current_y > line_position) or (prev_y > line_position and current_y <= line_position):
                if object_id in previous_centroids:
                    prev_centroid, prev_time = previous_centroids[object_id]
                    distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                    time_elapsed = current_time - prev_time
                    speed = distance / time_elapsed

                    total_speed += speed
                    count_people += 1
                    cv2.putText(frame, f"{speed:.2f} px/sec", (centroid[0] - 20, centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        previous_y[object_id] = current_y
        previous_centroids[object_id] = (centroid, current_time)

    if count_people > 0:
        average_speed = total_speed / count_people
        if average_speed > 250:
            crowd_behavior = "Crowd is aggressive and fast"
        elif 150 <= average_speed <= 250:
            crowd_behavior = "Crowd is normal"
        else:
            crowd_behavior = "Crowd is too slow"

        cv2.putText(frame, crowd_behavior, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if total_frames - frame_number <= fps:
        cv2.putText(frame, crowd_behavior, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Final Crowd Behavior", frame)
        cv2.waitKey(1000)
        break
    
    cv2.line(frame, (0, line_position), (w, line_position), (0, 0, 255), 2)
    cv2.putText(frame, f"Up: {up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Down: {down_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("People Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
