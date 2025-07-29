import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import time
import os
import pandas as pd
import plotly.express as px
import soundfile as sf
import pyaudio
import mediapipe as mp

# Import your project's modules
from emotion_detecton import detect_emotion, guess_reason, detect_voice_emotion
from sign_language import recognize_sign_language_webcam # Updated import to reflect changes
from privacy import clear_session_data
from diary import log_emotion, get_diary_data

# --- Global Configuration ---
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30
EMOTION_DETECTION_INTERVAL = 0.8 # Seconds between face emotion detections
SIGN_DETECTION_INTERVAL = 0.5    # Seconds between sign language detections

AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_ANALYSIS_SECONDS = 3 # Analyze audio every 3 seconds

# --- Video Capture and Processing Thread ---
class VideoCaptureThread(QThread):
    frame_processed = pyqtSignal(np.ndarray, str, str, object, object, object)

    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.last_emotion_time = 0
        self.last_sign_time = 0

        self.mp_hands = mp.solutions.hands
        self.hands = None # MediaPipe hands object, initialized/cleaned based on mode
        self.mp_drawing = mp.solutions.drawing_utils

        self.current_mode = "none" # "emotion", "sign", or "none"

    def set_mode(self, mode):
        """Sets the detection mode for the thread."""
        if self.current_mode != mode: # Only update if mode changes
            print(f"Video thread mode set to: {mode}")
            self.current_mode = mode
            # If switching from sign to emotion, ensure hands is cleaned up
            if mode == "emotion" and self.hands is not None:
                self._cleanup_mediapipe()
            # If switching from emotion to sign, (re)initialize hands if needed
            elif mode == "sign" and self.hands is None and self.running:
                # Re-initialize hands if it's currently None and we are running
                self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            # If switching to 'none' (e.g., stopping webcam), clean up hands
            elif mode == "none" and self.hands is not None:
                self._cleanup_mediapipe()


    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0) # 0 for default webcam

        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            self.running = False
            self._cleanup_mediapipe() # Ensure hands is cleaned even if cap fails
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS)

        # Initialize MediaPipe hands object ONLY if we are in 'sign' mode initially
        if self.current_mode == "sign" and self.hands is None:
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        print(f"INFO: Video stream started in mode: {self.current_mode}.")

        # Variables to hold detection results
        current_emotion_text = "No face detected."
        current_face_bbox = None
        current_sign_text = "No hand detected."
        current_hand_landmarks_results_obj = None # Raw MediaPipe results for drawing

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Convert the raw BGR frame to RGB for processing with DeepFace/MediaPipe
            # and for consistent display with QImage.Format_RGB888
            processed_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # The frame to display will be this RGB version, on which detections will be drawn
            display_frame = processed_frame_rgb.copy() 
            current_time = time.time()

            # --- Conditional Detection Logic based on current_mode ---
            if self.current_mode == "emotion":
                # Only perform Emotion Detection
                if current_time - self.last_emotion_time > EMOTION_DETECTION_INTERVAL:
                    # Pass the RGB frame to detect_emotion
                    emotion_data, deepface_error, face_detected, bounding_box = detect_emotion(processed_frame_rgb)
                    self.last_emotion_time = current_time
                    if face_detected and emotion_data and isinstance(emotion_data, dict) and 'emotion' in emotion_data:
                        detected_emotion = max(emotion_data['emotion'], key=emotion_data['emotion'].get)
                        confidence = emotion_data['emotion'][detected_emotion]
                        current_emotion_text = f"Emotion: {detected_emotion.title()} ({confidence:.2f}%)"
                        current_face_bbox = bounding_box
                        log_emotion(detected_emotion, f"Webcam: {guess_reason(detected_emotion)}")
                    else:
                        current_emotion_text = "No face detected."
                        current_face_bbox = None
                        log_emotion("No_Detection", deepface_error if deepface_error else "No face found in webcam frame")
                
                # Draw face bounding box if detected on the RGB display_frame
                if current_face_bbox:
                    x, y, w_box, h_box = current_face_bbox
                    # Draw rectangle with RGB green color
                    cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    # Put text with RGB green color
                    cv2.putText(display_frame, current_emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Ensure sign detection related variables are reset/cleared if not in sign mode
                current_sign_text = "N/A (Emotion Mode)"
                current_hand_landmarks_results_obj = None


            elif self.current_mode == "sign":
                # Only perform Sign Language Detection
                # Ensure self.hands is initialized if not already (e.g., if switching modes)
                if self.hands is None: # Should ideally be initialized by set_mode or at run start
                    self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

                if current_time - self.last_sign_time > SIGN_DETECTION_INTERVAL:
                    # Pass the RGB frame to recognize_sign_language_webcam
                    # NOTE: recognize_sign_language_webcam now only returns 3 values
                    recognized_sign, hand_detected, mp_results = recognize_sign_language_webcam(processed_frame_rgb)
                    
                    self.last_sign_time = current_time
                    if hand_detected:
                        current_sign_text = f"Sign: {recognized_sign.title()}" # Removed Emotion part
                        current_hand_landmarks_results_obj = mp_results
                        # Removed log_emotion here, as it's not hand emotion based now
                    else:
                        current_sign_text = "No hand detected."
                        current_hand_landmarks_results_obj = None
                        # Removed log_emotion here
                elif not self.hands: # If hands wasn't initialized, display a message
                    current_sign_text = "No hand detected. (MediaPipe not ready)"

                # Drawing hand landmarks happens in the SignLanguagePage.update_frame_slot
                # The raw_mp_results object is passed for that purpose.
                # However, any custom drawing on display_frame here must use RGB colors.

                # Ensure emotion detection related variables are reset/cleared if not in emotion mode
                current_emotion_text = "N/A" # No longer "N/A (Sign Mode)"
                current_face_bbox = None

            else: # self.current_mode == "none" or stopped
                # When mode is 'none', show generic messages
                current_emotion_text = "Webcam Paused/Stopped"
                current_face_bbox = None
                current_sign_text = "Webcam Paused/Stopped"
                current_hand_landmarks_results_obj = None


            # --- Diagnostic Drawing: Blue square (now that display_frame is RGB) ---
            # This little blue square confirms the video stream is active and frames are updating
            if display_frame.shape[0] > 10 and display_frame.shape[1] > 10:
                display_frame[5:15, 5:15] = (0, 0, 255) # Blue square for RGB frame

            # Emit the processed frame and detection results (some may be N/A or None based on mode)
            self.frame_processed.emit(display_frame, current_emotion_text, current_sign_text, current_face_bbox, current_hand_landmarks_results_obj, current_hand_landmarks_results_obj)

            QThread.msleep(int(1000 / WEBCAM_FPS)) # Control frame rate

        # Cleanup operations after the loop finishes
        if self.cap:
            self.cap.release()
            self.cap = None # Set to None after releasing

        self._cleanup_mediapipe() # Ensure hands is closed on thread exit
        print(f"INFO: Video stream stopped and released. Final mode: {self.current_mode}.")

    def _cleanup_mediapipe(self):
        """Ensures MediaPipe hands object is closed only once."""
        if self.hands is not None:
            try:
                self.hands.close()
                print("INFO: MediaPipe hands object closed.")
            except Exception as e:
                print(f"WARNING: Error closing MediaPipe hands object: {e}")
            finally:
                self.hands = None # Crucial: Set to None after attempting to close

    def stop(self):
        """Signals the thread to stop and waits for its termination."""
        self.running = False
        self.wait()


# --- Audio Capture and Processing Thread ---
class AudioCaptureThread(QThread):
    voice_emotion_detected = pyqtSignal(str, str) # Emits (emotion, reason)
    status_message = pyqtSignal(str) # Emits microphone status updates

    def __init__(self):
        super().__init__()
        self.running = False
        self.p = None # PyAudio instance
        self.stream = None # Audio stream
        self.audio_buffer = [] # Buffer for audio chunks

    def run(self):
        self.running = True
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=AUDIO_FORMAT,
                                        channels=AUDIO_CHANNELS,
                                        rate=AUDIO_RATE,
                                        input=True,
                                        frames_per_buffer=AUDIO_CHUNK_SIZE)
            self.status_message.emit("Microphone status: Listening...")
            print("INFO: Audio stream started.")

            while self.running:
                try:
                    # Read audio data from the stream
                    # exception_on_overflow=False prevents crashes if buffer overflows
                    data = self.stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                    self.audio_buffer.append(data)

                    # Check if enough audio is collected for analysis
                    total_samples_in_buffer = len(self.audio_buffer) * AUDIO_CHUNK_SIZE
                    if total_samples_in_buffer >= AUDIO_ANALYSIS_SECONDS * AUDIO_RATE:
                        # Concatenate chunks and convert to numpy array
                        audio_data_raw = b''.join(self.audio_buffer)
                        audio_data_np = np.frombuffer(audio_data_raw, dtype=np.int16)

                        temp_wav_path = "temp_voice_segment.wav"
                        try:
                            # Save to a temporary WAV file for deepface analysis
                            sf.write(temp_wav_path, audio_data_np, AUDIO_RATE)

                            # Perform voice emotion detection
                            emotion, confidence = detect_voice_emotion(temp_wav_path)
                            reason = guess_reason(emotion)
                            self.voice_emotion_detected.emit(emotion, reason)
                            log_emotion(emotion, f"Real-time Voice: {reason}")
                        except Exception as e:
                            print(f"Error during real-time voice emotion detection: {e}")
                            self.voice_emotion_detected.emit("Error", f"Analysis failed: {e}")
                            self.status_message.emit("Microphone status: Error during analysis.")
                        finally:
                            # Clean up the temporary WAV file
                            if os.path.exists(temp_wav_path):
                                os.remove(temp_wav_path)

                        self.audio_buffer = [] # Clear buffer for next segment

                except IOError as e:
                    # Handle specific audio device errors
                    print(f"IOError in audio stream: {e}")
                    self.voice_emotion_detected.emit("Error", "Audio device error.")
                    self.status_message.emit("Microphone status: Device Error! Check microphone.")
                    self.running = False # Stop the loop on critical error
                    break
                except Exception as e:
                    # Catch any other unexpected errors
                    print(f"Unexpected error in audio stream: {e}")
                    self.voice_emotion_detected.emit("Error", f"Unexpected error: {e}")
                    self.status_message.emit("Microphone status: Unexpected Error!")
                    self.running = False # Stop the loop on critical error
                    break

            self._cleanup_audio_stream() # Ensure cleanup when loop exits

        except Exception as e:
            # Handle errors during PyAudio initialization or stream opening
            print(f"Failed to initialize/open audio stream: {e}")
            self.status_message.emit(f"Microphone status: Init Error! {e}. Check PyAudio/microphone setup.")
            self._cleanup_audio_stream()

    def _cleanup_audio_stream(self):
        """Closes the audio stream and terminates PyAudio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.p:
            self.p.terminate()
            self.p = None
        print("INFO: Audio stream stopped and cleaned up.")

    def stop(self):
        """Signals the thread to stop and waits for its termination."""
        self.running = False
        self.wait()


# --- UI Pages Classes ---

class EmotionDetectionPage(QWidget):
    def __init__(self, parent_main_window):
        super().__init__()
        self.parent_main_window = parent_main_window
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.info_label = QLabel("Real-time webcam feed for emotion detection. Look directly at the camera.")
        self.info_label.setFont(QFont("Arial", 10))
        self.layout.addWidget(self.info_label)

        self.video_label = QLabel("Webcam Feed - Emotion Detection")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(WEBCAM_WIDTH, WEBCAM_HEIGHT)
        self.video_label.setStyleSheet("border: 1px solid grey; background-color: black;")
        self.layout.addWidget(self.video_label)

        self.emotion_status_label = QLabel("Emotion: Not Detecting")
        self.emotion_status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(self.emotion_status_label)

        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Webcam")
        self.start_button.clicked.connect(self.start_webcam)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Webcam")
        self.stop_button.clicked.connect(self.stop_webcam)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        self.layout.addLayout(control_layout)

        self.layout.addStretch()

    @pyqtSlot()
    def start_webcam(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Stop audio thread if it's running
        self.parent_main_window.voice_page.stop_realtime_audio()

        # Set video thread mode to "emotion"
        self.parent_main_window.video_thread.set_mode("emotion")

        if not self.parent_main_window.video_thread.isRunning():
            self.parent_main_window.video_thread.start()
        self.emotion_status_label.setText("Webcam active. Detecting emotion...")

    @pyqtSlot()
    def stop_webcam(self):
        self.parent_main_window.video_thread.stop()
        # Set video thread mode to "none" when stopped
        self.parent_main_window.video_thread.set_mode("none") 
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.setText("Webcam Feed Stopped")
        self.emotion_status_label.setText("Emotion: Not Detecting")

    @pyqtSlot(np.ndarray, str, str, object, object, object)
    def update_frame_slot(self, frame_data, emotion_text, sign_text, face_bbox, hand_landmarks_results_obj, raw_mp_results):
        # Only update if this page is currently visible
        if self.parent_main_window.pages.currentWidget() == self:
            h, w, ch = frame_data.shape
            bytes_per_line = ch * w
            # frame_data is now RGB, so use Format_RGB888 and no rgbSwapped()
            q_image = QImage(frame_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image) 
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Only display emotion text received
            self.emotion_status_label.setText(emotion_text)


class SignLanguagePage(QWidget):
    def __init__(self, parent_main_window):
        super().__init__()
        self.parent_main_window = parent_main_window
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.info_label = QLabel("Real-time webcam feed for sign language recognition. Show your hand gestures.")
        self.info_label.setFont(QFont("Arial", 10))
        self.layout.addWidget(self.info_label)

        self.video_label = QLabel("Webcam Feed - Sign Language")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(WEBCAM_WIDTH, WEBCAM_HEIGHT)
        self.video_label.setStyleSheet("border: 1px solid grey; background-color: black;")
        self.layout.addWidget(self.video_label)

        self.sign_status_label = QLabel("Sign: Not Detecting")
        self.sign_status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(self.sign_status_label)

        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Webcam")
        self.start_button.clicked.connect(self.start_webcam)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Webcam")
        self.stop_button.clicked.connect(self.stop_webcam)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        self.layout.addLayout(control_layout)

        self.layout.addStretch()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands # Used for HAND_CONNECTIONS constant

    @pyqtSlot()
    def start_webcam(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Stop audio thread if it's running
        self.parent_main_window.voice_page.stop_realtime_audio()

        # Set video thread mode to "sign"
        self.parent_main_window.video_thread.set_mode("sign")

        if not self.parent_main_window.video_thread.isRunning():
            self.parent_main_window.video_thread.start()
        self.sign_status_label.setText("Webcam active. Detecting sign...")

    @pyqtSlot()
    def stop_webcam(self):
        self.parent_main_window.video_thread.stop()
        # Set video thread mode to "none" when stopped
        self.parent_main_window.video_thread.set_mode("none") 
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.setText("Webcam Feed Stopped")
        self.sign_status_label.setText("Sign: Not Detecting")

    @pyqtSlot(np.ndarray, str, str, object, object, object)
    def update_frame_slot(self, frame_data, emotion_text, sign_text, face_bbox, hand_landmarks_results_obj, raw_mp_results):
        # Only update if this page is currently visible
        if self.parent_main_window.pages.currentWidget() == self:
            frame_to_display = np.copy(frame_data) # Make a writable copy for drawing (now RGB)

            # Draw hand landmarks if present (raw_mp_results passed by thread)
            if raw_mp_results and raw_mp_results.multi_hand_landmarks:
                for hand_landmarks in raw_mp_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame_to_display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Optionally draw bounding box around hands
                    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_min = int(min(x_coords) * frame_to_display.shape[1])
                    x_max = int(max(x_coords) * frame_to_display.shape[1])
                    y_min = int(min(y_coords) * frame_to_display.shape[0])
                    y_max = int(max(y_coords) * frame_to_display.shape[0])
                    
                    padding = 20
                    # Use RGB Yellow (255, 255, 0)
                    cv2.rectangle(frame_to_display, (max(0, x_min - padding), max(0, y_min - padding)),
                                  (min(frame_to_display.shape[1], x_max + padding), min(frame_to_display.shape[0], y_max + padding)),
                                  (255, 255, 0), 2)

            h, w, ch = frame_to_display.shape
            bytes_per_line = ch * w
            # frame_to_display is now RGB, so use Format_RGB888 and no rgbSwapped()
            q_image = QImage(frame_to_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image) 
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Only display sign text received (emotion part removed)
            self.sign_status_label.setText(sign_text)


class VoiceEmotionPage(QWidget):
    def __init__(self, parent_main_window):
        super().__init__()
        self.parent_main_window = parent_main_window
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel("Voice Emotion Detection")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.layout.addWidget(self.title_label)

        # --- Audio File Upload Section ---
        self.file_upload_group_title = QLabel("Analyze an Audio File:")
        self.file_upload_group_title.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.file_upload_group_title)

        self.file_upload_layout = QHBoxLayout()
        self.upload_button = QPushButton("Browse Audio File (WAV/MP3)")
        self.upload_button.clicked.connect(self.upload_audio)
        self.file_upload_layout.addWidget(self.upload_button)

        self.audio_file_status = QLabel("No file selected.")
        self.file_upload_layout.addWidget(self.audio_file_status)
        self.layout.addLayout(self.file_upload_layout)

        self.file_result_label = QLabel("File Emotion: N/A")
        font_italic = QFont("Arial", 12)
        font_italic.setItalic(True) # Use setItalic for QFont
        self.file_result_label.setFont(font_italic)
        self.layout.addWidget(self.file_result_label)

        self.layout.addSpacing(20)

        # --- Real-time Audio Section ---
        self.realtime_group_title = QLabel("Real-time Microphone Input:")
        self.realtime_group_title.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.realtime_group_title)

        self.realtime_buttons_layout = QHBoxLayout()
        self.start_mic_button = QPushButton("Start Real-time Mic")
        self.start_mic_button.clicked.connect(self.start_realtime_audio)
        self.realtime_buttons_layout.addWidget(self.start_mic_button)

        self.stop_mic_button = QPushButton("Stop Real-time Mic")
        self.stop_mic_button.clicked.connect(self.stop_realtime_audio)
        self.stop_mic_button.setEnabled(False)
        self.realtime_buttons_layout.addWidget(self.stop_mic_button)
        self.layout.addLayout(self.realtime_buttons_layout)

        self.realtime_status_label = QLabel("Microphone status: Idle")
        self.layout.addWidget(self.realtime_status_label)

        self.realtime_emotion_label = QLabel("Real-time Emotion: N/A")
        self.realtime_emotion_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(self.realtime_emotion_label)

        self.realtime_reason_label = QLabel("Real-time Reason: N/A")
        self.layout.addWidget(self.realtime_reason_label)

        self.layout.addStretch()

        # Connect audio thread signals
        self.parent_main_window.audio_thread.voice_emotion_detected.connect(self.update_realtime_emotion)
        self.parent_main_window.audio_thread.status_message.connect(self.update_realtime_status)


    @pyqtSlot()
    def upload_audio(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3);;All Files (*)")

        if file_path:
            self.audio_file_status.setText(f"Loaded: {os.path.basename(file_path)}")
            try:
                emotion, confidence = detect_voice_emotion(file_path)
                reason = guess_reason(emotion)
                self.file_result_label.setText(f"File Emotion: {emotion.title()} ({confidence*100:.2f}%)")
                log_emotion(emotion, f"Voice: {reason}")
                QMessageBox.information(self, "Success", "Audio file processed and emotion logged!")
            except Exception as e:
                self.file_result_label.setText("Error during file analysis.")
                QMessageBox.critical(self, "Error", f"Failed to detect voice emotion from file: {str(e)}")
        else:
            self.audio_file_status.setText("No file selected.")
            self.file_result_label.setText("File Emotion: N/A")

    @pyqtSlot()
    def start_realtime_audio(self):
        # Stop video thread if it's running when starting audio
        if self.parent_main_window.video_thread.isRunning():
            self.parent_main_window.video_thread.stop()
            self.parent_main_window.video_thread.set_mode("none") # Set mode to none
            self.parent_main_window.emotion_page.start_button.setEnabled(True)
            self.parent_main_window.emotion_page.stop_button.setEnabled(False)
            self.parent_main_window.sign_page.start_button.setEnabled(True)
            self.parent_main_window.sign_page.stop_button.setEnabled(False)
            self.parent_main_window.emotion_page.video_label.setText("Webcam Feed Stopped (Switched to Voice)")
            self.parent_main_window.sign_page.video_label.setText("Webcam Feed Stopped (Switched to Voice)")


        if not self.parent_main_window.audio_thread.isRunning():
            self.parent_main_window.audio_thread.start()
            self.start_mic_button.setEnabled(False)
            self.stop_mic_button.setEnabled(True)
            self.realtime_emotion_label.setText("Real-time Emotion: Detecting...")
            self.realtime_reason_label.setText("Real-time Reason: N/A")
        else:
            self.realtime_status_label.setText("Microphone status: Already listening.")

    @pyqtSlot()
    def stop_realtime_audio(self):
        if self.parent_main_window.audio_thread.isRunning():
            self.parent_main_window.audio_thread.stop()
        self.start_mic_button.setEnabled(True)
        self.stop_mic_button.setEnabled(False)
        self.realtime_status_label.setText("Microphone status: Stopped.")
        self.realtime_emotion_label.setText("Real-time Emotion: N/A")
        self.realtime_reason_label.setText("Real-time Reason: N/A")

    @pyqtSlot(str, str)
    def update_realtime_emotion(self, emotion, reason):
        # Only update if this page is currently visible
        if self.parent_main_window.pages.currentWidget() == self:
            self.realtime_emotion_label.setText(f"Real-time Emotion: {emotion.title()}")
            self.realtime_reason_label.setText(f"Real-time Reason: {reason}")

    @pyqtSlot(str)
    def update_realtime_status(self, message):
        # Only update if this page is currently visible
        if self.parent_main_window.pages.currentWidget() == self:
            self.realtime_status_label.setText(message)


class DiaryPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel("Emotion Diary")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.layout.addWidget(self.title_label)

        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers) # Make table read-only
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows) # Select entire rows
        self.table_widget.setAlternatingRowColors(True) # For better readability
        self.layout.addWidget(self.table_widget)

        self.plot_label = QLabel("Emotion Timeline Plot (Loading...)")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_label.setMinimumSize(600, 250) # Set a minimum size for the plot area
        self.layout.addWidget(self.plot_label)

        os.makedirs('session_data', exist_ok=True) # Ensure session_data directory exists
        # Data is loaded when page is displayed, not just initialized

        self.layout.addStretch()

    @pyqtSlot()
    def load_diary_data(self):
        df = get_diary_data() # Get data from diary.py

        if not df.empty:
            self.table_widget.setRowCount(df.shape[0])
            self.table_widget.setColumnCount(df.shape[1])
            self.table_widget.setHorizontalHeaderLabels(df.columns)
            
            for row_idx, row_data in df.iterrows():
                for col_idx, item in enumerate(row_data):
                    self.table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(item)))
            
            self.table_widget.horizontalHeader().setStretchLastSection(True)
            self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

            try:
                # Generate Plotly plot and save as image
                df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure timestamp is datetime
                # Use a specific order for emotions if desired, otherwise Plotly will sort alphabetically
                emotion_order = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'No_Detection']
                
                fig = px.scatter(df, x='timestamp', y='emotion', color='emotion',
                                 category_orders={"emotion": emotion_order}, # Apply custom order
                                 title='Emotion Timeline', labels={'emotion': 'Emotion', 'timestamp': 'Time'})
                
                plot_image_path = "temp_plot.png"
                # Save plot as PNG using Kaleido engine
                fig.write_image(plot_image_path, width=self.plot_label.width(), height=self.plot_label.height())

                # Load the saved image into QLabel
                pixmap = QPixmap(plot_image_path)
                self.plot_label.setPixmap(pixmap.scaled(self.plot_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.plot_label.setText("") # Clear "Loading..." text
                os.remove(plot_image_path) # Clean up temporary file

            except ImportError:
                # Catch specific ImportError for kaleido
                self.plot_label.setText("Error: 'kaleido' package not found. Install with 'pip install -U kaleido' to enable plotting.")
                QMessageBox.warning(self, "Plot Error", "The 'kaleido' package is not installed. Plotly charts cannot be exported to images. Please install it: 'pip install -U kaleido'.")
            except Exception as e:
                self.plot_label.setText(f"Error loading plot: {e}")
                print(f"Plotly Error: {e}")
                QMessageBox.warning(self, "Plot Error", f"Could not generate emotion timeline plot: {e}.")

        else:
            self.table_widget.setRowCount(0)
            self.table_widget.clearContents()
            self.table_widget.setHorizontalHeaderLabels(["timestamp", "emotion", "reason"])
            self.plot_label.setText("No diary entries to display yet.")


class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel("Settings and Privacy")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.layout.addWidget(self.title_label)

        self.privacy_info_label = QLabel("Manage your privacy settings and clear data.")
        self.layout.addWidget(self.privacy_info_label)

        self.clear_data_button = QPushButton("Clear All Session Data")
        self.clear_data_button.clicked.connect(self.clear_data)
        self.layout.addWidget(self.clear_data_button)

        self.layout.addStretch()

    @pyqtSlot()
    def clear_data(self):
        reply = QMessageBox.question(self, 'Clear Data',
                                     "Are you sure you want to clear all session data (including diary entries)? This action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            clear_session_data()
            QMessageBox.information(self, "Success", "All session data has been cleared!")


# --- Main Application Window ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion & Sign Language Desktop App")
        self.setGeometry(100, 100, 1000, 700) # (x, y, width, height)

        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # Sidebar for navigation
        self.sidebar = QListWidget()
        self.sidebar.setMaximumWidth(180)
        self.sidebar.setFont(QFont("Arial", 12))
        self.sidebar.addItem("Detect Emotion")
        self.sidebar.addItem("Sign Language")
        self.sidebar.addItem("Voice Emotion")
        self.sidebar.addItem("Diary")
        self.sidebar.addItem("Settings")
        self.sidebar.currentRowChanged.connect(self.display_page) # Connect sidebar clicks to page display
        self.main_layout.addWidget(self.sidebar)

        # Content Area for different pages
        self.pages = QStackedWidget()
        self.main_layout.addWidget(self.pages)

        # Initialize threads BEFORE page instances that use them
        self.video_thread = VideoCaptureThread()
        self.audio_thread = AudioCaptureThread()

        # Create instances of pages, passing self (MainWindow) if they need to interact with threads
        self.emotion_page = EmotionDetectionPage(self)
        self.sign_page = SignLanguagePage(self)
        self.voice_page = VoiceEmotionPage(self)
        self.diary_page = DiaryPage()
        self.settings_page = SettingsPage()

        # Add pages to the stacked widget
        self.pages.addWidget(self.emotion_page)
        self.pages.addWidget(self.sign_page)
        self.pages.addWidget(self.voice_page)
        self.pages.addWidget(self.diary_page)
        self.pages.addWidget(self.settings_page)

        # Connect video thread signals to relevant page slots
        self.video_thread.frame_processed.connect(self.emotion_page.update_frame_slot)
        self.video_thread.frame_processed.connect(self.sign_page.update_frame_slot)

        # Set initial page and mode
        self.sidebar.setCurrentRow(0) # Select first item (Emotion Detection)
        self.video_thread.set_mode("emotion") # Initialize video thread mode


    @pyqtSlot(int)
    def display_page(self, index):
        # Stop webcam if it's running when switching from a video-related page
        if self.video_thread.isRunning():
            self.video_thread.stop()
            # Reset UI elements on the video pages
            self.emotion_page.video_label.setText("Webcam Feed Stopped")
            self.sign_page.video_label.setText("Webcam Feed Stopped")
            self.emotion_page.start_button.setEnabled(True)
            self.emotion_page.stop_button.setEnabled(False)
            self.sign_page.start_button.setEnabled(True)
            self.sign_page.stop_button.setEnabled(False)
        
        # Stop audio stream if it's running when switching from Voice Emotion page
        if self.pages.currentWidget() == self.voice_page and self.audio_thread.isRunning():
            self.voice_page.stop_realtime_audio()
            
        self.pages.setCurrentIndex(index)
        
        # Set video thread mode based on the selected page
        if index == 0: # Emotion Detection Page
            self.video_thread.set_mode("emotion")
        elif index == 1: # Sign Language Page
            self.video_thread.set_mode("sign")
        else: # Other pages (Voice, Diary, Settings), set mode to 'none' for video thread
            self.video_thread.set_mode("none")

        # Reload diary data if the Diary page is selected
        if index == 3: # Index of Diary page (0-indexed)
            self.diary_page.load_diary_data()

    def closeEvent(self, event):
        # Ensure ALL threads are stopped when the main window is closed
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.set_mode("none") # Final cleanup for mode
        if self.audio_thread.isRunning():
            self.audio_thread.stop()
        event.accept() # Accept the close event


# --- Main Application Entry Point ---
class MainApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("Emotion AI Desktop")

if __name__ == "__main__":
    app = MainApplication(sys.argv)
    window = MainWindow() # Instantiate MainWindow at the top level
    window.show() # Show the main window
    sys.exit(app.exec_()) # Start the PyQt event loop