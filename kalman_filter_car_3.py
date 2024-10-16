import cv2
import numpy as np

# Initialize Kalman Filter for a 6D state (x, y, vx, vy, flow_x, flow_y)
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0, 0],
                                    [0, 1, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01

# Load the video
cap = cv2.VideoCapture('carro_amarillo.mp4')

# Read the first frame for optical flow
ret, first_frame = cap.read()
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect yellow color
    lower_y = np.array([20, 100, 120])  # Lower bound for yellow 20, 50, 70
    upper_y = np.array([32, 255, 255])  # Upper bound for yellow 32, 255, 255
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_y = cv2.inRange(hsvFrame, lower_y, upper_y)

    # Output for yellow detection (element 1 for Kalman filter)
    output_y = cv2.bitwise_and(frame, frame, mask=mask_y)

    # Find contours in the yellow mask to determine the centroid
    contours, _ = cv2.findContours(mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_centroid = None

    if contours:
        # Find the largest contour (assuming the yellow object is the largest yellow region)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            yellow_centroid = (cx, cy)

    # Optical Flow Calculation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(first_frame_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute the magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Get the mean position of moving pixels (optional but can help reduce noise)
    moving_pixels = np.column_stack(np.where(magnitude > 2))  # Use a threshold for moving pixels
    moving_centroid = None
    if moving_pixels.size > 0:
        moving_centroid = np.mean(moving_pixels, axis=0).astype(int)

    # Create a mask to visualize the optical flow
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    result = cv2.addWeighted(frame, 1, flow_image, 2, 0)

    # Convert to grayscale to detect circles
    gray = cv2.cvtColor(output_y, cv2.COLOR_BGR2GRAY)
    rows = gray.shape[0]

    # Detect circles (element 2 for Kalman filter)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=rows/8, param1=80, param2=30, minRadius=3, maxRadius=30)

    # Create a separate frame for showing HoughCircles
    hough_frame = frame.copy()

    measured = None  # Initialize variable for the measured position

    # Draw the detected circles on the separate HoughCircles frame
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(hough_frame, center, radius, (100, 0, 255), 3)  # Draw outline of the circle in pink
            cv2.circle(hough_frame, center, 3, (100, 0, 255), -1)  # Draw center of the circle in green

            # Use the circle center for measurement if detected
            if yellow_centroid is not None:
                # Combine yellow centroid and circle center (simple average)
                measured_x = int((yellow_centroid[0] + center[0]) / 2)
                measured_y = int((yellow_centroid[1] + center[1]) / 2)
            else:
                measured_x, measured_y = center[0], center[1]

            measured = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
    
    elif yellow_centroid is not None:
        # Use the yellow centroid if no circles are detected
        measured = np.array([[np.float32(yellow_centroid[0])], [np.float32(yellow_centroid[1])]])
        measured_x, measured_y = yellow_centroid

    # Incorporate optical flow centroid into measurement
    if moving_centroid is not None:
        measured_x = int((measured_x + moving_centroid[0]) / 2)
        measured_y = int((measured_y + moving_centroid[1]) / 2)
        measured = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])

    # Kalman Filter: Prediction step
    prediction = kalman.predict()

    # Kalman Filter: Correction step if measurement is available
    if measured is not None:
        kalman.correct(measured)
        measured_x, measured_y = int(measured[0]), int(measured[1])

    # Draw the predicted position from Kalman Filter on the original frame (only Kalman output)
    predicted_x, predicted_y = int(prediction[0]), int(prediction[1])
    kalman_frame = frame.copy()
    cv2.circle(kalman_frame, (predicted_x, predicted_y), 5, (0, 255, 0), -1)  # Green for predicted position

    # Display windows for HoughCircles, Yellow mask, Kalman filter, and Optical Flow
    cv2.imshow('HoughCircles', hough_frame)  # Shows the HoughCircles detected
    cv2.imshow('Yellow Detection', output_y)  # Shows the mask for yellow detection
    cv2.imshow('Kalman Filter Output', kalman_frame)  # Shows only the Kalman filter output
    cv2.imshow('Optical Flow', result)  # Shows the optical flow visualization

    # Update the first frame for the next iteration
    first_frame_gray = gray_frame.copy()

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

