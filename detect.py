import jetson.inference
import jetson.utils
import cv2
import numpy as np

# Load model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Camera
camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput()

font = cv2.FONT_HERSHEY_SIMPLEX

while display.IsStreaming():
    img = camera.Capture()
    detections = net.Detect(img)

    # Convert to OpenCV
    frame = jetson.utils.cudaToNumpy(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    anomaly = False

    for det in detections:
        class_name = net.GetClassDesc(det.ClassID)

        if class_name == "person":
            x1, y1 = int(det.Left), int(det.Top)
            x2, y2 = int(det.Right), int(det.Bottom)

            width = x2 - x1
            height = y2 - y1

            # Simple posture logic
            if width > height:
                posture = "LYING"
                anomaly = True
            elif height > width * 1.5:
                posture = "STANDING"
                anomaly = False
            else:
                posture = "SITTING"
                anomaly = True

            # Color
            color = (0,0,255) if anomaly else (0,255,0)

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)

            # Label
            cv2.putText(frame, posture, (x1,y1-10),
                        font, 0.7, color, 2)

    # -------- IRON MAN UI --------
    cv2.putText(frame, "JARVIS SYSTEM ONLINE", (30,40),
                font, 0.8, (0,0,255), 2)

    cv2.putText(frame, "SCANNING TARGET...", (30,80),
                font, 0.6, (0,255,255), 2)

    if anomaly:
        cv2.putText(frame, "⚠ THREAT DETECTED", (30,120),
                    font, 0.8, (0,0,255), 3)
        cv2.putText(frame, "STATUS: INACTIVE", (30,160),
                    font, 0.7, (0,0,255), 2)
    else:
        cv2.putText(frame, "SYSTEM NORMAL", (30,120),
                    font, 0.8, (0,255,0), 2)

    # Show window
    cv2.imshow("IronMan Anomaly Detection", frame)

    if cv2.waitKey(1) == 27:
        break

