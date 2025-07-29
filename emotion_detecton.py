import cv2
from deepface import DeepFace
import pandas as pd
import numpy as np
import os
import tensorflow as tf # Import tensorflow for suppressing warnings
import logging # For DeepFace's internal logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging (1=INFO, 2=WARNING, 3=ERROR)
tf.get_logger().setLevel('ERROR') # Set TensorFlow logger to only show ERRORs

# Suppress DeepFace warnings/info logs - make sure to do this before any DeepFace calls
# DeepFace uses Python's logging module
logging.getLogger('deepface').setLevel(logging.WARNING) # Only show warnings and errors from DeepFace

# --- Emotion Detection for Face (using DeepFace) ---
def detect_emotion(frame):
    """
    Detects emotion from a given image frame using DeepFace.
    The input frame is expected to be in RGB format.
    """
    face_detected = False
    emotion_data = None
    deepface_error = None
    bounding_box = None

    # Ensure frame is not empty
    if frame is None or frame.size == 0:
        return None, "Empty frame received", False, None

    try:
        # DeepFace expects RGB images
        # The frame passed from desktop_app.py is already RGB
        
        # DeepFace.analyze can detect faces automatically
        # Set enforce_detection=False to avoid errors if no face is detected,
        # but it will return an empty list or raise an error if no face is found and enforce_detection=True.
        # It's better to catch the exception.
        
        # models=['Emotion'] ensures only emotion model is loaded/used.
        # detector_backend='opencv' is generally reliable and fast.
        demographies = DeepFace.analyze(
            img_path=frame, 
            actions=['emotion'], 
            enforce_detection=False, # Allow no face to be detected without error
            detector_backend='opencv' # You can try 'ssd', 'dlib', 'mtcnn', 'retinaface' if 'opencv' struggles
        )

        if demographies: # Check if the list of detections is not empty
            # DeepFace returns a list of dictionaries, one for each face found.
            # We assume the largest face or the first one is the primary subject.
            # For simplicity, we'll take the first detected face.
            face_data = demographies[0] # Get the first detected face's data

            if 'emotion' in face_data:
                emotion_data = face_data # Store the entire face_data dictionary
                face_detected = True
                
                # Extract bounding box (facial_area comes from DeepFace analysis)
                if 'facial_area' in face_data:
                    x = face_data['facial_area']['x']
                    y = face_data['facial_area']['y']
                    w = face_data['facial_area']['w']
                    h = face_data['facial_area']['h']
                    bounding_box = (x, y, w, h)
                else:
                    # Fallback if facial_area is not directly available, though it should be
                    print("WARNING: 'facial_area' not found in DeepFace output.")
            else:
                deepface_error = "Emotion data not found in DeepFace output."
        else:
            deepface_error = "No face detected by DeepFace."

    except Exception as e:
        deepface_error = f"DeepFace error: {str(e)}"
        # print(f"DEBUG: DeepFace error during analysis: {e}") # Keep this for debugging if needed

    return emotion_data, deepface_error, face_detected, bounding_box

# --- Voice Emotion Detection ---
# Requires speech_recognition, pydub, and deepface for audio
# Ensure you have installed: pip install SpeechRecognition pydub deepface soundfile
# Also requires ffmpeg for pydub to handle mp3. Install ffmpeg and add to PATH.

def detect_voice_emotion(audio_path):
    """
    Detects emotion from an audio file using DeepFace's audio analysis.
    audio_path: Path to the audio file (WAV, MP3 etc.)
    Returns: (emotion, confidence)
    """
    # DeepFace's audio analysis handles its own loading.
    try:
        # DeepFace's audio module also uses its own models
        # It expects a path to the audio file.
        obj = DeepFace.analyze(
            img_path=audio_path, # img_path is used for both image and audio paths by DeepFace
            actions=['audio'],
            enforce_detection=False # For audio, this might refer to segments, usually not critical.
        )
        
        # obj will be a list of dictionaries, one per segment/speaker if multiple.
        # Assuming single speaker/segment for simplicity for now.
        if obj and isinstance(obj, list) and len(obj) > 0 and 'emotion' in obj[0]:
            emotions = obj[0]['emotion'] # This is a dictionary of emotions and scores
            
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            return dominant_emotion, confidence
        else:
            return "No_Detection", 0.0 # No emotion detected or unexpected output
            
    except Exception as e:
        print(f"Error in detect_voice_emotion: {e}")
        return "Error", 0.0 # Return error status

# --- Reason Guessing ---
def guess_reason(emotion):
    """Provides a simple, generic reason based on the detected emotion."""
    emotion = emotion.lower()
    if emotion == 'happy':
        return "You seem content and joyful."
    elif emotion == 'sad':
        return "You may be feeling down or reflective."
    elif emotion == 'angry':
        return "You appear to be feeling frustrated or irritated."
    elif emotion == 'surprise':
        return "You seem astonished or excited."
    elif emotion == 'fear':
        return "You may be experiencing some apprehension."
    elif emotion == 'disgust':
        return "You seem to be feeling aversion or disapproval."
    elif emotion == 'neutral':
        return "You seem calm and composed."
    else:
        return "Emotion detection in progress or not clear."

if __name__ == '__main__':
    # This block is for testing emotion_detection.py independently

    # Test Face Emotion Detection
    print("--- Testing Face Emotion Detection ---")
    # You need an image file for this test, or a webcam feed
    # For a quick test, you can use a sample image or enable webcam for a moment.

    # Example: Using a dummy black image (will report no face)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) # RGB black image
    emotion_data, error, detected, bbox = detect_emotion(dummy_frame)
    print(f"Test 1 (No Face): Detected: {detected}, Emotion: {emotion_data}, Error: {error}")

    # Example: If you have an image file (e.g., 'sample_face.jpg')
    # Make sure 'sample_face.jpg' is in the same directory or provide full path
    # try:
    #     sample_image = cv2.imread('sample_face.jpg')
    #     if sample_image is not None:
    #         sample_image_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB) # Convert to RGB for DeepFace
    #         emotion_data, error, detected, bbox = detect_emotion(sample_image_rgb)
    #         print(f"Test 2 (Sample Image): Detected: {detected}, Emotion: {emotion_data}, BBox: {bbox}, Error: {error}")
    #     else:
    #         print("Could not load sample_face.jpg. Skipping image test.")
    # except Exception as e:
    #     print(f"Error loading sample image for test: {e}")


    # Test Voice Emotion Detection
    print("\n--- Testing Voice Emotion Detection ---")
    # You need an audio file for this test. Create a short 'test_audio.wav' or 'test_audio.mp3'
    # For example, record your voice saying "hello" for 3-5 seconds and save as test_audio.wav
    test_audio_path = "test_audio.wav" # Replace with your test audio file
    if os.path.exists(test_audio_path):
        print(f"Analyzing {test_audio_path}...")
        emotion, confidence = detect_voice_emotion(test_audio_path)
        reason = guess_reason(emotion)
        print(f"Voice Emotion: {emotion.title()} (Confidence: {confidence*100:.2f}%) - Reason: {reason}")
    else:
        print(f"Skipping voice test: '{test_audio_path}' not found.")
        print("Please create a 'test_audio.wav' file for voice emotion testing.")

    print("\n--- Emotion Detection Tests Complete ---")