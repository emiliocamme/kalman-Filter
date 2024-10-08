import cv2
import numpy as np

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Load the video
cap = cv2.VideoCapture('kalman.mp4')

# Loop through each frame in the video
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # Exit the loop if no frames are returned (end of the video)

    # Resize the frame for faster processing (optional)
    frame_resized = cv2.resize(frame, (800, 640))
    cropped_image = frame_resized[150:500, 0:800]

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Blur the frame to reduce noise
    blurred_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)


    # Define the red color range in HSV (two ranges for different shades of red)
    lower_red = np.array([0, 35,0])
    upper_red = np.array([6,149,172])
    mask_red = cv2.inRange(blurred_frame, lower_red, upper_red)

    # Noise removal
    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(mask_red, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variable to store the center of the detected contour
    measured = None

    # Loop through each contour
    for contour in contours:
        
        area = cv2.contourArea(contour)#The function computes a contour area. Similarly to moments , the area is computed using the Green formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using drawContours or fillPoly , can be different.
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) # approximates a curve or a polygon with another curve/polygon with less vertices so that the distance between them is less or equal to the specified precision. It uses the Douglas-Peucker algorithm
        

        if area > 650 and area < 950:  # Filter small areas
            #cv2.drawContours(cropped_image, [approx], 0, (255, 0, 0), 2)  # Draw in blue for visibility
            # Calculate the moments of the contour
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + w // 2, y + h // 2 #center
                measured = np.array([[np.float32(cx)], [np.float32(cy)]])  # Measurement for Kalman filter
                cv2.circle(cropped_image, (cx, cy), 8, (0, 165, 255), -1)  # Radius 8, orange
    
    # Kalman Filter: Prediction step
    prediction = kalman.predict()#the Kalman filter estimates the next position of the object based on the previous state (position and velocity). This prediction helps track the object even when it temporarily disappears or is occluded

    if measured is not None:
        # Kalman Filter: Correction step if a measurement is available
        kalman.correct(measured)
        measured_x, measured_y = int(measured[0]), int(measured[1])
        cv2.circle(cropped_image, (measured_x, measured_y), 3, (100, 0, 255), -1)  # mini
    else:
        # Use predicted value if no measurement is available (i.e., object is temporarily lost)
        measured_x, measured_y = None, None

    # Draw the predicted position from the Kalman Filter
    predicted_x, predicted_y = int(prediction[0]), int(prediction[1])
    cv2.circle(cropped_image, (predicted_x, predicted_y), 5, (0, 255, 0), -1)  # green
    # Display the resulting frame with detections


    cv2.imshow('normal', frame_resized)
    
    cv2.imshow('Mask for Red (Lower)', mask_red)

    cv2.imshow('Eroded', eroded)

    cv2.imshow('Phone Detection', cropped_image)

    # Press 'q' to quit the video processing
    if cv2.waitKey(120) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
