from ultralytics import YOLO
import cv2
import os

# -------- Step 1: Load the model --------
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt for better accuracy

# -------- Step 2: Get user-defined limit --------
THRESHOLD = int(input("Enter max allowed crowd limit: "))

# -------- Step 3: Open webcam --------
cap = cv2.VideoCapture(0)  # Use video file path if not using webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- Step 4: Run YOLO on frame --------
    results = model(frame, stream=True)
    people_count = 0

    # -------- Step 5: Count 'person' objects only --------
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if model.names[int(cls)] == "person":
                people_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -------- Step 6: Show count on screen --------
    text = f"People Count: {people_count}"
    if people_count > THRESHOLD:
        text += " ⚠️ ALERT!"
        os.system('say "Crowd limit exceeded"')  # Works on macOS

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Crowd Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
