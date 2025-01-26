import streamlit as st

import numpy as np

import pandas as pd

import time

from datetime import datetime

from collections import deque

import mediapipe as mp

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from scipy.spatial.distance import euclidean

class PupilTracker(VideoTransformerBase):

    def __init__(self, fixation_threshold=30, fixation_duration=100):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(

            max_num_faces=1,

            refine_landmarks=True,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5

        )

        self.LEFT_IRIS = [474, 475, 476, 477]

        self.RIGHT_IRIS = [469, 470, 471, 472]

        self.fixation_threshold = fixation_threshold

        self.fixation_duration = fixation_duration

        self.positions_history = deque(maxlen=10)

        self.current_fixation = None

        self.fixations = []

        self.tracking_data = []

    def transform(self, frame):

        frame = frame.to_ndarray(format="bgr24")

        frame, left_center, right_center = self.detect_pupil(frame)

        if left_center is not None and right_center is not None:

            cv2.circle(frame, tuple(left_center), 3, (0, 255, 0), -1)

            cv2.circle(frame, tuple(right_center), 3, (0, 255, 0), -1)

            current_time = time.time() * 1000

            avg_position = np.mean([left_center, right_center], axis=0)

            fixation = self.detect_fixation(avg_position, current_time)

            if fixation is not None:

                cv2.circle(frame, tuple(fixation['position'].astype(int)), 10, (0, 0, 255), 2)

        return frame

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

                    'position': np.mean(positions, axis=0),

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

    # User inputs for configuration

    fixation_threshold = st.slider("Fixation Threshold (pixels)", 10, 100, 30)

    fixation_duration = st.slider("Fixation Duration (milliseconds)", 50, 500, 100)

    # Initialize session state for tracker and recording state

    if 'tracker' not in st.session_state:

        st.session_state.tracker = PupilTracker(fixation_threshold, fixation_duration)

    if 'recording' not in st.session_state:

        st.session_state.recording = False

    # Control buttons

    col1, col2 = st.columns(2)

    start_button = col1.button("Start Recording")

    stop_button = col2.button("Stop Recording")

    if start_button:

        st.session_state.recording = True

        st.session_state.tracker.tracking_data = []

    if stop_button:

        st.session_state.recording = False

        if len(st.session_state.tracker.tracking_data) > 0:

            df = pd.DataFrame(st.session_state.tracker.tracking_data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"pupil_tracking_data_{timestamp}.csv"

            df.to_csv(filename, index=False)

            st.success(f"Data saved to {filename}")

    # Streamlit WebRTC component for video capture

    webrtc_streamer(key="example", video_transformer_factory=lambda: st.session_state.tracker)

if __name__ == "__main__":

    main()
