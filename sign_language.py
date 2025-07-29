import cv2
import mediapipe as mp
import numpy as np
import joblib # For loading scikit-learn models
import os

# --- MediaPipe Setup (must match training script) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Model Paths ---
MODEL_DIR = os.path.join(os.path.dirname(__file__)) # Directory of the current script
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Global variables for loaded model and scaler
svm_model = None
scaler = None
SIGNS = [] # This will be populated from the model's classes

# --- Feature Extraction Function (MUST BE IDENTICAL TO train_sign_model.py) ---
def extract_hand_features(landmarks):
    """
    Extracts a flattened array of hand landmark coordinates and distances.
    Input: MediaPipe HandLandmarks object.
    Output: A 1D numpy array of features.
    """
    if not landmarks:
        # Return a zero array of expected length if no landmarks
        # This length (21*3 + 20) must match what your scaler was trained on.
        return np.zeros(21 * 3 + 20)

    features = []
    # 1. Landmark Coordinates (x, y, z for each of 21 landmarks)
    for landmark in landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])

    # 2. Distances from wrist (landmark 0) to all other landmarks (1-20)
    # This provides a scale-invariant feature and improves robustness.
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    for i in range(1, 21):
        point = np.array([landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z])
        features.append(np.linalg.norm(wrist - point)) # Euclidean distance

    expected_len = 21 * 3 + 20 # 21 (x,y,z) + 20 distances
    if len(features) != expected_len:
        # This warning is crucial for debugging feature extraction mismatches
        print(f"WARNING: Feature vector length mismatch! Expected {expected_len}, got {len(features)}.")
        # Attempt to adjust, but ideally, this indicates an error in extraction logic
        if len(features) < expected_len:
            features.extend([0.0] * (expected_len - len(features)))
        elif len(features) > expected_len:
            features = features[:expected_len]

    return np.array(features)

# --- Model Loading ---
def load_sign_language_model():
    global svm_model, scaler, SIGNS
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print("WARNING: Sign language model files not found:")
            print(f"  Model Path: {MODEL_PATH}")
            print(f"  Scaler Path: {SCALER_PATH}")
            print("Please train your sign language model using train_sign_model.py and save it as svm_model.pkl and scaler.pkl.")
            svm_model = None # Ensure they are None if not found
            scaler = None
            SIGNS = []
            return False

        if svm_model is None: # Only load if not already loaded
            svm_model = joblib.load(MODEL_PATH)
            print(f"INFO: Loaded SVM model from {MODEL_PATH}")
        if scaler is None: # Only load if not already loaded
            scaler = joblib.load(SCALER_PATH)
            print(f"INFO: Loaded scaler from {SCALER_PATH}")
        
        SIGNS = list(svm_model.classes_) # Get the class labels from the trained model
        print(f"INFO: Model trained for signs: {SIGNS}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load sign language model or scaler: {e}")
        print("Please ensure your .pkl files are valid and compatible with the current joblib version.")
        svm_model = None
        scaler = None
        SIGNS = []
        return False

# Ensure model is loaded when the module is imported
# This makes it ready for recognize_sign_language_webcam calls immediately.
load_sign_language_model()

# --- Real-time Sign Language Recognition ---
def recognize_sign_language_webcam(frame_rgb):
    """
    Recognizes sign language from a single webcam frame.
    This function expects the frame to be RGB.
    It internally uses MediaPipe to detect hands and then an SVM model for prediction.
    
    Input: frame_rgb (np.ndarray) - An RGB image frame.
    Output: (recognized_sign_text, hand_detected_boolean, mediapipe_results_object)
    """
    global svm_model, scaler, SIGNS
    
    recognized_sign = "No_Prediction"
    hand_detected = False
    mp_results = None

    # Load model if it wasn't loaded at import time (e.g., if files were missing initially)
    if svm_model is None or scaler is None:
        if not load_sign_language_model(): # Try to load if not already loaded
            # If model still can't be loaded, return default values and indicate issue
            return "Model_Not_Loaded", False, None 

    # Create a hands object for processing within this function
    # NOTE: desktop_app.py's VideoCaptureThread already manages an mp_hands.Hands object.
    # To avoid creating multiple instances (which can be inefficient), the ideal
    # approach is to pass the 'results' directly from VideoCaptureThread after it
    # processes the frame.
    # For now, let's keep it self-contained for robustness and testing, but be aware.
    
    # We will simulate the hands.process call directly here for robustness.
    # In desktop_app.py, VideoCaptureThread already calls hands.process and passes
    # raw_mp_results to the update_frame_slot. This function is called by the thread
    # to *get* the sign prediction, not to re-process MediaPipe.
    #
    # To fix the "too many values to unpack", we need to ensure this function
    # ONLY returns 3 values as expected by desktop_app.py.
    
    # Let's adjust this to accept the already-processed MediaPipe results if available.
    # However, the desktop_app.py currently passes the 'processed_frame_rgb'
    # so we must run MediaPipe here.

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_processor:
        results = hands_processor.process(frame_rgb)
        mp_results = results # Store results for potential drawing in UI

        if results.multi_hand_landmarks:
            hand_detected = True
            # For simplicity, assume one hand and process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Sign Language Prediction
            features = extract_hand_features(hand_landmarks)
            
            # Ensure features are a 2D array for scaler.transform
            features_reshaped = features.reshape(1, -1) 
            
            try:
                scaled_features = scaler.transform(features_reshaped)
                prediction = svm_model.predict(scaled_features)[0]
                recognized_sign = str(prediction) # Convert to string to avoid numpy.str_ issues
                
            except Exception as e:
                recognized_sign = "Prediction_Error"
                print(f"ERROR: Sign language prediction failed: {e}")
        else:
            recognized_sign = "No_Hand_Detected" # Update status when no hand is found

    return recognized_sign, hand_detected, mp_results

# --- Standalone Test (only runs when sign_language.py is executed directly) ---
if __name__ == '__main__':
    print("--- Testing Sign Language Recognition Module ---")
    
    # Simple webcam test for sign recognition
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam for test.")
    else:
        print("Webcam open. Show signs. Press 'q' to quit.")
        
        # Initialize MediaPipe Hands for the standalone test
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands_test:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
                
                frame = cv2.flip(frame, 1) # Flip for mirror effect

                # Convert to RGB for MediaPipe processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe Hands
                results_test = hands_test.process(frame_rgb) # Use results_test to avoid conflict
                
                # Call the main recognition function with the RGB frame
                # This ensures we are testing the function as it would be called by desktop_app.py
                recognized_sign_text, hand_was_detected, mp_results_for_drawing = recognize_sign_language_webcam(frame_rgb)
                
                # Draw landmarks and bounding box based on mp_results_for_drawing
                if mp_results_for_drawing and mp_results_for_drawing.multi_hand_landmarks:
                    for hand_landmarks in mp_results_for_drawing.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Draw bounding box (optional, for visualization)
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        h_frame, w_frame, _ = frame.shape
                        x_min, x_max = int(min(x_coords) * w_frame), int(max(x_coords) * w_frame)
                        y_min, y_max = int(min(y_coords) * h_frame), int(max(y_coords) * h_frame)
                        padding = 20
                        cv2.rectangle(frame, (max(0, x_min - padding), max(0, y_min - padding)),
                                    (min(w_frame, x_max + padding), min(h_frame, y_max + padding)),
                                    (0, 255, 255), 2) # Yellow in BGR

                cv2.putText(frame, f"Sign: {recognized_sign_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Convert back to BGR for cv2.imshow
                frame_bgr_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Use the original RGB frame for display conversion
                # Overlay drawing done on 'frame', which is BGR. So just show it.
                cv2.imshow('Sign Language Test', frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
    print("--- Sign Language Test Complete ---")