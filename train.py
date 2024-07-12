import cv2
import numpy as np

# Initialize parameters for line detection
prev_frame = None
prev_edges = None
prev_left_lane = None
prev_right_lane = None

# Open video capture object
cap = cv2.VideoCapture('video/test2.mp4')

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while(cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Cut away invalid frame area (if any)
    height, width = frame.shape[:2]
    frame = frame[int(height/3):, :]

    # Make a copy of the original frame
    original_frame = frame.copy()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Sliding window properties initialization
    window_height = 20
    window_width = 40
    margin = 100
    min_pixels = 50
    n_windows = 9

    # Bezier curve variables initialization
    left_points = []
    right_points = []

    # Iterate over sliding windows
    for window in range(n_windows):
        # Define window boundaries
        y_low = height - (window + 1) * window_height
        y_high = height - window * window_height

        # Find edges within the window for left side
        left_window = edges[y_low:y_high, :int(width/2)]
        if np.any(left_window):
            x_left_low = np.max([np.min(np.where(left_window != 0)[1]) - margin, 0])
            x_left_high = np.min([np.max(np.where(left_window != 0)[1]) + margin, int(width/2)])
            left_points.append((x_left_low, y_low))
            left_points.append((x_left_high, y_high))

        # Find edges within the window for right side
        right_window = edges[y_low:y_high, int(width/2):]
        if np.any(right_window):
            x_right_low = np.max([np.min(np.where(right_window != 0)[1]) - margin, 0]) + int(width/2)
            x_right_high = np.min([np.max(np.where(right_window != 0)[1]) + margin, width])
            right_points.append((x_right_low, y_low))
            right_points.append((x_right_high, y_high))

    # Process left railroad line
    if len(left_points) > 0:
        prev_left_lane = left_points
    elif prev_left_lane is not None:
        left_points = prev_left_lane

    # Process right railroad line
    if len(right_points) > 0:
        prev_right_lane = right_points
    elif prev_right_lane is not None:
        right_points = prev_right_lane

    # Bezier curve processing
    if len(left_points) > 0 and len(right_points) > 0:
        # Fit bezier curve for left line
        left_points = np.array(left_points)
        left_coefficients = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
        left_curve_y = np.linspace(0, height, height)
        left_curve_x = left_coefficients[0] * left_curve_y ** 2 + left_coefficients[1] * left_curve_y + left_coefficients[2]

        # Fit bezier curve for right line
        right_points = np.array(right_points)
        right_coefficients = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
        right_curve_y = np.linspace(0, height, height)
        right_curve_x = right_coefficients[0] * right_curve_y ** 2 + right_coefficients[1] * right_curve_y + right_coefficients[2]

        # Draw bezier curves on original frame
        for i in range(len(left_curve_x) - 1):
            cv2.line(original_frame, (int(left_curve_x[i]), int(left_curve_y[i])), (int(left_curve_x[i + 1]), int(left_curve_y[i + 1])), (0, 255, 0), 5)
        for i in range(len(right_curve_x) - 1):
            cv2.line(original_frame, (int(right_curve_x[i]), int(right_curve_y[i])), (int(right_curve_x[i + 1]), int(right_curve_y[i + 1])), (0, 255, 0), 5)

    # Display the original frame with lines drawn on the track
    cv2.imshow('Railway Track with Lines', original_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
