import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
from scipy.spatial.distance import euclidean
from collections import deque
import mediapipe as mp
from dyslexia_model import DyslexiaDetector

class PupilTracker:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Pupil landmarks indices for both eyes
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # Parameters for fixation detection
        self.fixation_threshold = 20  # pixels
        self.fixation_duration = 40  # milliseconds
        self.positions_history = deque(maxlen=10)
        self.current_fixation = None
        self.fixations = []

        # Data storage
        self.tracking_data = []

    def detect_pupil(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [frame.shape[1], frame.shape[0]]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])

            # Get iris landmarks
            left_iris = mesh_points[self.LEFT_IRIS]
            right_iris = mesh_points[self.RIGHT_IRIS]

            # Calculate iris centers
            left_center = np.mean(left_iris, axis=0).astype(int)
            right_center = np.mean(right_iris, axis=0).astype(int)

            return frame, left_center, right_center

        return frame, None, None

    def detect_fixation(self, current_position, timestamp):
        self.positions_history.append((current_position, timestamp))

        if len(self.positions_history) < 2:
            return None

        # Check if current positions form a fixation
        positions = np.array([p[0] for p in self.positions_history])
        timestamps = np.array([p[1] for p in self.positions_history])

        max_distance = max([euclidean(positions[0], p) for p in positions])
        duration = timestamps[-1] - timestamps[0]

        if max_distance < self.fixation_threshold and duration >= self.fixation_duration:
            if self.current_fixation is None:
                self.current_fixation = {
                    'start_time': timestamps[0],
                    'position': np.mean(positions, axis=0) if len(positions) > 0 else None,
                    'duration': duration
                }
            else:
                self.current_fixation['duration'] = duration
        else:
            if self.current_fixation is not None:
                self.fixations.append(self.current_fixation)
                self.current_fixation = None

        return self.current_fixation

def sample_text():
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font" onclick="openFullscreen()">Please Read The Following: </p>', unsafe_allow_html=True)

    st.write("""
        Little Pepík preferred to stay at home reading books full of adventures 
        rather than playing outside with other children. Perhaps this was also 
        influenced by his injured leg, which prevented him from much mischief. 
        Instead, he immersed himself in stories full of sneaky Indians, brave 
        pioneers, or mischievous pirates. Occasionally, he would sadly look out
        the window at his friends playing and watch their lively games from afar.         
        
        One day, while he was, as usual, partially hidden behind the curtain, he
        noticed that some shadow was moving like a flash up and down the tree. A 
        small, brown, and very agile squirrel was darting around the tree, searching
        for nuts to store in its secret hideaway.
    """)

def open_full_screen_text():
    st.markdown("""
    <script>
    function openNewWindow() {
        var newWindow = window.open("", "_blank", "width=800,height=600");
        newWindow.document.write(`
            <html>
            <head>
                <title>Full Screen Text</title>
                <style>
                    body {
                        font-size: 20px;
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        background-color: #f4f4f4;
                    }
                    .content {
                        width: 80%;
                        text-align: justify;
                        overflow: auto;
                    }
                </style>
            </head>
            <body>
                <div class="content">
                    <h2>Please Read The Following:</h2>
                    <p>Little Pepík preferred to stay at home reading books full of adventures rather than playing outside with other children. Perhaps this was also influenced by his injured leg, which prevented him from much mischief. Instead, he immersed himself in stories full of sneaky Indians, brave pioneers, or mischievous pirates. Occasionally, he would sadly look out the window at his friends playing and watch their lively games from afar. One day, while he was, as usual, partially hidden behind the curtain, he noticed that some shadow was moving like a flash up and down the tree. A small, brown, and very agile squirrel was darting around the tree, searching for nuts to store in its secret hideaway.</p>
                </div>
            </body>
            </html>
        `);
        newWindow.document.close();
    }
    openNewWindow();
    </script>
    """, unsafe_allow_html=True)

def main():
    st.title("Advanced Pupil Tracker")

    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = DyslexiaDetector()

    if 'tracker' not in st.session_state:
        st.session_state.tracker = PupilTracker()

    if 'recording' not in st.session_state:
        st.session_state.recording = False

    if 'text_displayed' not in st.session_state:
        st.session_state.text_displayed = False

    reading_content_button = st.button("Reading Content")

    uploaded_file = st.file_uploader("STEM_Fair_Data", type=["png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded Image.', use_column_width=True)
    # Camera input
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    text_placeholder = st.empty()  # Placeholder for sample text

    # Control buttons
    col1, col2 = st.columns(2)
    start_button = col1.button("Start Recording")
    stop_button = col2.button("Stop Recording")

    # Full-screen button
    fullscreen_button = st.button("Show Text in Full-Screen")  # New full-screen button

    if fullscreen_button:
        open_full_screen_text()  # Open the full-screen window with text

    if start_button:
        st.session_state.recording = True
        st.session_state.tracker.tracking_data = []

        # Show sample text only once
        if not st.session_state.text_displayed:
            text_placeholder.empty()
            sample_text()
            st.session_state.text_displayed = True

    if stop_button:
        st.session_state.recording = False
        st.session_state.text_displayed = False  # Reset the state
        text_placeholder.empty()  # Remove the sample text

        # Save data
        if len(st.session_state.tracker.tracking_data) > 0:
             # Prepare fixation data
            df = pd.DataFrame(st.session_state.tracker.tracking_data)

            fixation_data = df[df['is_fixation']]

            fixation_data['duration'] = fixation_data['fixation_duration']

            fixation_data['x'] = fixation_data['fixation_x']

            fixation_data['y'] = fixation_data['fixation_y']

            prediction, probability = st.session_state.detector.predict(fixation_data)

             # Display results

            st.header("Dyslexia Detection Results")

            if prediction is not None:

                if len(probability) > 1:

                    st.warning(f"Indicators of dyslexia detected (Confidence: {probability[1]:.2f})")

                else:

                    st.warning("Prediction made, but confidence levels are not available.")

            else:

                st.success(f"No indicators of dyslexia detected (Confidence: {probability[0]:.2f})")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pupil_tracking_data_{timestamp}.csv"
            df.to_csv(filename, index=False)
            st.success(f"Data saved to {filename}")

    try:
        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame, left_center, right_center = st.session_state.tracker.detect_pupil(frame)

            if left_center is not None and right_center is not None:
                # Draw pupils
                cv2.circle(frame, tuple(left_center), 3, (0, 255, 0), -1)
                cv2.circle(frame, tuple(right_center), 3, (0, 255, 0), -1)

                # Detect fixations
                current_time = time.time() * 1000  # Convert to milliseconds
                avg_position = np.mean([left_center, right_center], axis=0)

                fixation = st.session_state.tracker.detect_fixation(avg_position, current_time)

                # Record data if recording is active
                if st.session_state.recording:
                    data_point = {
                        'timestamp': current_time,
                        'left_pupil_x': left_center[0],
                        'left_pupil_y': left_center[1],
                        'right_pupil_x': right_center[0],
                        'right_pupil_y': right_center[1],
                        'is_fixation': fixation is not None
                    }
                    if fixation is not None:
                        data_point.update({
                            'fixation_duration': fixation['duration'],
                            'fixation_x': fixation['position'][0] if fixation['position'] is not None else None,
                            'fixation_y': fixation['position'][1] if fixation['position'] is not None else None
                        })
                    st.session_state.tracker.tracking_data.append(data_point)

                # Draw fixation
                if fixation is not None and fixation['position'] is not None:
                    cv2.circle(frame, tuple(fixation['position'].astype(int)),
                               10, (0, 0, 255), 2)

            # Display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

    except Exception as e:
        st.error(f"Error: {str(e)}")

    finally:
        cap.release()

if __name__ == "__main__":
    main()
