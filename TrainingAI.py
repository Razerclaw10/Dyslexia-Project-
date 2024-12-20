'''
import streamlit as st
import cv2
import os

# Function to record the webcam
def record_webcam(filename="recording.avi", fps=20.0, resolution=(640, 480)):
    """Records video from the default webcam and saves it to a file."""
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    st.write("Recording in progress... Press 'Finish' to stop recording.")
    frame_placeholder = st.empty()

    while st.session_state.recording:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        out.write(frame)
        frame_placeholder.image(frame, channels="BGR", clamp=True)

        st.experimental_rerun()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.success(f"Recording saved as {filename}")

# Function to display text for the user to read
def sample_text():
    if "paragraph" not in st.session_state:
        st.session_state.paragraph = 0

    if st.session_state.paragraph == 0:
        st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">Please Read The Following: </p>', unsafe_allow_html=True)
        st.write("""
            The cat sat on the mat. Yesterday, it was raining—pouring, actually! 
            Does the small, striped feline like to play? Sometimes, the mat is red, but other times it’s blue. 
            Surprisingly, the cat didn't care. She simply stared at the puddles. Puddles, puddles, and more puddles! 
            “Why are there so many?” thought the curious cat. What would you think if you saw puddles everywhere?
        """)

        # Finish button stops the recording and retains the paragraph state
        if st.button('Finish Reading'):
            st.session_state.paragraph = 1
            st.session_state.recording = False

    if st.session_state.paragraph > 0:
        st.write("Thank you for reading!")

# Initialize session state for recording
if "recording" not in st.session_state:
    st.session_state.recording = False

# Main UI
if st.button("Start Recording"):
    st.session_state.recording = True

if st.session_state.recording:
    record_webcam()

sample_text()
'''
import streamlit as st
import cv2
import os
import time

# Function to record the webcam
def record_webcam(filename="recording.avi", fps=20.0, resolution=(640, 480)):
    """Records video from the default webcam and saves it to a file."""
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    st.write("Recording in progress... Press 'Finish Reading' to stop.")
    frame_placeholder = st.empty()

    while st.session_state.recording:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        out.write(frame)
        frame_placeholder.image(frame, channels="BGR", clamp=True)

        # Small delay to reduce CPU usage
        time.sleep(0.05)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.success(f"Recording saved as {filename}")

# Function to display text for the user to read
def sample_text():
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Please Read The Following: </p>', unsafe_allow_html=True)
    st.write("""
        The cat sat on the mat. Yesterday, it was raining—pouring, actually! 
        Does the small, striped feline like to play? Sometimes, the mat is red, but other times it’s blue. 
        Surprisingly, the cat didn't care. She simply stared at the puddles. Puddles, puddles, and more puddles! 
        “Why are there so many?” thought the curious cat. What would you think if you saw puddles everywhere?
    """)

    if st.button('Finish Reading'):
        st.session_state.recording = False

# Initialize session state for recording
if "recording" not in st.session_state:
    st.session_state.recording = False

if st.button("Start Recording"):
    st.session_state.recording = True

if st.session_state.recording:
    sample_text()
    record_webcam()