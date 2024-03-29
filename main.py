import cv2

# Replace 'path/to/your/video.mp4' with the actual path to your video file
video_path = 'path/to/your/video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and display each frame of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("End of video.")
        break

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
