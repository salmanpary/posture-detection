import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set flag to draw the output image with annotations
        draw = True

        # Process the image and detect the pose
        results = pose.process(image)

        # Draw the pose landmarks on the image
        if draw and results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between the shoulders, hips, and knees
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            # Convert the landmark positions to pixel coordinates
            image_height, image_width, _ = image.shape
            left_shoulder_px = int(left_shoulder.x * image_width)
            left_shoulder_py = int(left_shoulder.y * image_height)
            right_shoulder_px = int(right_shoulder.x * image_width)
            right_shoulder_py = int(right_shoulder.y * image_height)
            left_hip_px = int(left_hip.x * image_width)
            left_hip_py = int(left_hip.y * image_height)
            right_hip_px = int(right_hip.x * image_width)
            right_hip_py = int(right_hip.y * image_height)
            left_knee_px = int(left_knee.x * image_width)
            left_knee_py = int(left_knee.y * image_height)
            right_knee_px = int(right_knee.x * image_width)
            right_knee_py = int(right_knee.y * image_height)

            # Calculate the angle between the shoulders, hips, and knees
            angle_shoulders = abs(right_shoulder_py - left_shoulder_py)
            angle_hips = abs(right_hip_py - left_hip_py)
            angle_knees = abs(right_knee_py - left_knee_py)

            # Check if the person's posture is straight
            if angle_shoulders <= 50 and angle_hips <= 50 and angle_knees <= 50:
                cv2.putText(image, "Posture Correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Posture Not Correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release