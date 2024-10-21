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

kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03

# Load the video
cap = cv2.VideoCapture('carro_amarillo.mp4')

# Parameters for ShiTomasi corner detection for Optical Flow
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first frame and find corners for optical flow
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing optical flow tracks
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect yellow color
    lower_y = np.array([20, 100, 120])  # Lower bound for yellow
    upper_y = np.array([32, 255, 255])  # Upper bound for yellow
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
            cv2.circle(hough_frame, center, 3, (100, 0, 255), -1)  # Draw center of the circle

            # Use the circle center for measurement if detected
            if yellow_centroid is not None:
                measured_x = int((yellow_centroid[0] + center[0]) / 2)
                measured_y = int((yellow_centroid[1] + center[1]) / 2)
            else:
                measured_x, measured_y = center[0], center[1]

            measured = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])

    elif yellow_centroid is not None:
        # Use the yellow centroid if no circles are detected
        measured = np.array([[np.float32(yellow_centroid[0])], [np.float32(yellow_centroid[1])]])
        measured_x, measured_y = yellow_centroid

    # Optical Flow Calculation
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points from the flow
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    flow_centroid = None
    if good_new.size > 0:
        flow_centroid = np.mean(good_new, axis=0).astype(int)

    # Update previous frame for next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Incorporate optical flow centroid into measurement
    if flow_centroid is not None:
        if measured is not None:
            measured_x = int((measured_x + flow_centroid[0]) / 2)
            measured_y = int((measured_y + flow_centroid[1]) / 2)
        else:
            measured_x, measured_y = flow_centroid

        measured = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])

    # Kalman Filter: Prediction step
    prediction = kalman.predict()

    # Kalman Filter: Correction step if measurement is available
    if measured is not None:
        kalman.correct(measured)
        measured_x, measured_y = int(measured[0]), int(measured[1])

    # Draw the predicted position from Kalman Filter on the original frame
    predicted_x, predicted_y = int(prediction[0]), int(prediction[1])
    kalman_frame = frame.copy()
    cv2.circle(kalman_frame, (predicted_x, predicted_y), 5, (0, 255, 0), -1)  # Green for predicted position

    # Draw optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    flow_img = cv2.add(frame, mask)

    # Display windows
    cv2.imshow('HoughCircles', hough_frame)  # Shows HoughCircles detected
    cv2.imshow('Yellow Detection', output_y)  # Shows yellow detection mask
    cv2.imshow('Kalman Filter Output', kalman_frame)  # Shows Kalman filter output
    cv2.imshow('Optical Flow', flow_img)  # Shows optical flow visualization

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

