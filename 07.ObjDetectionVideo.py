# pip install opencv-python 
import cv2

# Load the trained Haar cascade
object_cascade = cv2.CascadeClassifier("stop_data.xml")  # Replace with your cascade file

# Open the video capture
video_capture = cv2.VideoCapture("vid1.mp4")  # Or 0 for webcam

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Get the original frame dimensions
    frame_width = int(video_capture.get(3))  # Width
    frame_height = int(video_capture.get(4))  # Height

    # Desired display width (adjust as needed)
    display_width = 800  # Example: 800 pixels wide

    # Calculate the scaling factor to maintain aspect ratio
    scale_factor = display_width / frame_width

    # Calculate the new height based on the scaling factor
    display_height = int(frame_height * scale_factor)

    # Resize the frame
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # Detect objects in the frame
    objects = object_cascade.detectMultiScale(
        cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY),  # Use resized frame here
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(30 * scale_factor), int(30 * scale_factor))  # Scale minSize too!
    )

    for (x, y, w, h) in objects:
        # Scale the bounding box coordinates back to the original frame size if needed
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)
        original_w = int(w / scale_factor)
        original_h = int(h / scale_factor)

        # Draw the bounding box on the resized frame
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the resized frame with detections
    cv2.imshow("Object Detection", resized_frame)

    key = cv2.waitKey(1)  # Wait for 1 millisecond; returns the pressed key's ASCII value
    if key == ord('q'):  # Check if 'q' is pressed
        break  # Exit the loop
    elif key == 27:  # Check if 'esc' is pressed
        break  # Exit the loop
    elif cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
        break  # Exit the loop if the window is closed using the close button

# Release resources
video_capture.release()
cv2.destroyAllWindows()
