
import cv2

# Load video
cap = cv2.VideoCapture('dolphin.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Use background subtractor
    fgmask = fgbg.apply(hsv)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # filter small objects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display result
    cv2.imshow('Underwater Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()