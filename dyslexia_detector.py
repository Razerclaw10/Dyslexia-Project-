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

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

class DyslexiaDetector:

    def __init__(self, model_path='dyslexia_model.h5'):

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, fixation_data):

        # Select only required features

        features = fixation_data[['fixation_x', 'fixation_y', 'fixation_duration']].values

        # Reshape for single prediction if necessary

        if len(features.shape) == 1:

            features = features.reshape(1, -1)

        predictions = self.model.predict(features)

        return predictions

def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path)

    # Select only the required features

    features = df[['fixation_x', 'fixation_y', 'fixation_duration']]

    target = df['has_dyslexia']

    # Handle missing values

    features.fillna(0, inplace=True)

    scaler = StandardScaler()

    features_scaled = scaler.fit_transform(features)

    return train_test_split(features_scaled, target, test_size=0.2, random_state=42)

def create_model(input_shape):

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),  # 3 features

        tf.keras.layers.Dense(64, activation='relu'),
        
        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

class PupilTracker:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(

            max_num_faces=1,

            refine_landmarks=True,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5

        )

        self.LEFT_IRIS = [474, 475, 476, 477]

        self.RIGHT_IRIS = [469, 470, 471, 472]

        self.fixation_threshold = 20

        self.fixation_duration = 40

        self.positions_history = deque(maxlen=10)

        self.current_fixation = None

        self.fixations = []

        self.tracking_data = []

    def detect_pupil(self, frame):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:

            mesh_points = np.array([

                np.multiply([p.x, p.y], [frame.shape[1], frame.shape[0]]).astype(int)

                for p in results.multi_face_landmarks[0].landmark

            ])

            left_iris = mesh_points[self.LEFT_IRIS]

            right_iris = mesh_points[self.RIGHT_IRIS]

            left_center = np.mean(left_iris, axis=0).astype(int)

            right_center = np.mean(right_iris, axis=0).astype(int)

            return frame, left_center, right_center

        return frame, None, None

    def detect_fixation(self, current_position, timestamp):

        self.positions_history.append((current_position, timestamp))

        if len(self.positions_history) < 2:

            return None

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

def main():

    st.title("Advanced Pupil Tracker")

    # Load and preprocess the ETDD70 dataset

    X_train, X_val, y_train, y_val = load_and_preprocess_data('path_to_ETDD70_dataset.csv')

    # Create and train the model

    model = create_model((3,))

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    model.save('dyslexia_model.h5')

    # Initialize session state

    if 'detector' not in st.session_state:

        st.session_state.detector = DyslexiaDetector()

    if 'tracker' not in st.session_state:

        st.session_state.tracker = PupilTracker()

    if 'recording' not in st.session_state:

        st.session_state.recording = False

    if 'text_displayed' not in st.session_state:

        st.session_state.text_displayed = False

    # Camera input

    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    text_placeholder = st.empty()

    # Control buttons

    col1, col2 = st.columns(2)

    start_button = col1.button("Start Recording")

    stop_button = col2.button("Stop Recording")

    if start_button:

        time.sleep(3)

        st.session_state.recording = True

        st.session_state.tracker.tracking_data = []


    if stop_button:

        st.session_state.recording = False

        st.session_state.text_displayed = False

        text_placeholder.empty()

        if len(st.session_state.tracker.tracking_data) > 0:

            df = pd.DataFrame(st.session_state.tracker.tracking_data)

            fixation_data = df[df['is_fixation']]

            prediction = st.session_state.detector.predict(fixation_data)

            st.header("Dyslexia Detection Results")

            if prediction.mean() > 0.5:

                st.warning("Indicators of dyslexia detected.")

            else:

                st.success("No indicators of dyslexia detected.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"pupil_tracking_data_{timestamp}.csv"

            df.to_csv(filename, index=False)

            st.success(f"Data saved to {filename}")

    try:

        while st.session_state.recording:

            ret, frame = cap.read()

            if not ret:

                break

            frame, left_center, right_center = st.session_state.tracker.detect_pupil(frame)

            if left_center is not None and right_center is not None:

                cv2.circle(frame, tuple(left_center), 3, (0, 255, 0), -1)

                cv2.circle(frame, tuple(right_center), 3, (0, 255, 0), -1)

                current_time = time.time() * 1000

                avg_position = np.mean([left_center, right_center], axis=0)

                fixation = st.session_state.tracker.detect_fixation(avg_position, current_time)

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

                if fixation is not None and fixation['position'] is not None:

                    cv2.circle(frame, tuple(fixation['position'].astype(int)), 10, (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame, channels="RGB")

    except Exception as e:

        st.error(f"Error: {str(e)}")

    finally:

        cap.release()

if __name__ == "__main__":

    main()
