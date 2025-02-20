import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Compute the difference between frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours around moving objects
    for contour in contours:
        if cv2.contourArea(contour) < 700:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Motion Detector", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Press 'q' to exit
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
