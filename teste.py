import cv2
import mediapipe as mp
import requests

# Initialize MediaPipe Pose detection
mp_pose = mp.solutions.pose

# Open webcam
cap = cv2.VideoCapture(0)
line_left = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2 - 200)
line_right = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2 + 200)


state = None
# Initialize MediaPipe Pose detector
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detector:
    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = pose_detector.process(rgb_frame)
        height, width, _ = frame.shape
        #cx, cy = int(pose_landmark.x * width), int(pose_landmark.y * height)
        # Check if any poses are detected
        if results.pose_landmarks:
            for pose_landmark in results.pose_landmarks.landmark:
                # Obtém as coordenadas (x, y) do landmark
                height, width, _ = frame.shape
                cx, cy = int(pose_landmark.x * width), int(pose_landmark.y * height)

                if cx < line_left:
                    new_state = 'Entrou'
                    occurrence = 1
                elif cx > line_right:
                    new_state = 'Saiu'
                    occurrence = 0
                else:
                    new_state = None
                    occurrence = None

                # Verifica se houve uma mudança de estado
                if new_state and new_state != state:
                    print(new_state)
                    state = new_state

                    # Envie a solicitação HTTP com base no estado
                    try:
                        resp = requests.post("http://localhost:8080/record", json={"occurrence": occurrence, "room": "Laboratorio"})
                        print(resp)
                    except Exception as e:
                        print(f"Erro ao enviar a solicitação HTTP: {e}")
            # Render detected landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.line(frame, (line_left, 0), (line_left, height), (0, 255, 0), 2)
        cv2.line(frame, (line_right, 0), (line_right, height), (0, 0, 255), 2)
        cv2.imshow('Pose Detection', frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
