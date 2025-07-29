import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# --- Configuration ---
DATA_DIR = 'data'
IMG_DIR = os.path.join(DATA_DIR, 'images') # Not strictly used for saving images anymore, but good to keep structure
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')
MODEL_PATH = 'svm_model.pkl'
SCALER_PATH = 'scaler.pkl'

# --- Hand Landmark Model Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Data Collection Delay ---
COLLECTION_DELAY_SECONDS = 0.5 # Adjust this value (e.g., 0.5 for half a second, 1.0 for one second)

# --- Feature Extraction Function (MUST BE IDENTICAL to sign_language.py) ---
def extract_hand_features(landmarks):
    if not landmarks:
        return np.zeros(21 * 3 + 20)

    features = []
    for landmark in landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])

    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    for i in range(1, 21):
        point = np.array([landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z])
        features.append(np.linalg.norm(wrist - point))

    expected_len = 21 * 3 + 20
    if len(features) != expected_len:
        print(f"WARNING: Feature vector length mismatch! Expected {expected_len}, got {len(features)}.")
        if len(features) < expected_len:
            features.extend([0.0] * (expected_len - len(features)))
        elif len(features) > expected_len:
            features = features[:expected_len]
    return np.array(features)

# --- Data Collection Function ---
def collect_data(signs_to_collect):
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define all possible feature column names
    feature_column_names = [f'x{i}' for i in range(21)] + \
                           [f'y{i}' for i in range(21)] + \
                           [f'z{i}' for i in range(21)] + \
                           [f'dist_w{i}' for i in range(1, 21)]
    all_columns = feature_column_names + ['sign']

    # --- KEY CHANGE HERE: Load existing data or create new DataFrame ---
    if os.path.exists(FEATURES_FILE):
        print(f"INFO: Loading existing features from {FEATURES_FILE}...")
        try:
            # Ensure consistency of columns when loading
            existing_df = pd.read_csv(FEATURES_FILE)
            # Filter out any signs that are explicitly in signs_to_collect
            # This allows re-collecting a specific sign if desired
            signs_to_filter_out = [s.upper() for s in signs_to_collect]
            existing_df = existing_df[~existing_df['sign'].isin(signs_to_filter_out)]

            # Reindex to ensure all_columns are present, filling missing with NaN
            all_features_df = existing_df.reindex(columns=all_columns)

            print(f"INFO: Loaded {len(existing_df)} existing records for signs: {existing_df['sign'].unique().tolist()}")
        except Exception as e:
            print(f"WARNING: Could not load existing features file ({e}). Starting with empty DataFrame.")
            all_features_df = pd.DataFrame(columns=all_columns)
    else:
        print("INFO: No existing features file found. Starting with empty DataFrame.")
        all_features_df = pd.DataFrame(columns=all_columns)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        for sign in signs_to_collect:
            # Check if this sign already exists in the DataFrame and if we explicitly want to skip it
            # This logic allows for adding NEW signs only, or re-collecting if specified in input
            # If you want to ONLY add NEW signs and never re-collect, uncomment and adjust below
            # if sign.upper() in all_features_df['sign'].unique() and sign.upper() not in signs_to_collect_forcibly:
            #     print(f"INFO: Skipping '{sign.upper()}' as it already exists in the dataset. To re-collect, remove it from features.csv first or adjust logic.")
            #     continue

            print(f"\n--- Collecting data for: '{sign.upper()}' ---")
            print("Press 's' to start capturing. Press 'q' to skip to next sign.")
            
            frame_count = 0
            start_capture = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1) # Mirror effect
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = hands.process(frame_rgb)
                
                display_frame = frame.copy() # Use BGR frame for display

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        h, w, _ = display_frame.shape
                        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    if start_capture:
                        features = extract_hand_features(results.multi_hand_landmarks[0])
                        if features is not None and len(features) == len(feature_column_names):
                            feature_dict = dict(zip(feature_column_names, features))
                            feature_dict['sign'] = sign.upper() # Ensure sign is uppercase for consistency
                            
                            all_features_df = pd.concat([all_features_df, pd.DataFrame([feature_dict])], ignore_index=True)
                            frame_count += 1
                            cv2.putText(display_frame, f"Capturing: {sign.upper()} - {frame_count} frames", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            
                            time.sleep(COLLECTION_DELAY_SECONDS)
                        else:
                            cv2.putText(display_frame, "Invalid features extracted!", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                else:
                    cv2.putText(display_frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    if start_capture:
                        cv2.putText(display_frame, "Adjust hand to capture", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(display_frame, f"Sign: {sign.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if not start_capture:
                        print(f"Starting capture for '{sign.upper()}'...")
                    start_capture = True
                elif key == ord('q'):
                    print(f"Skipping '{sign.upper()}' with {frame_count} frames collected.")
                    break
                elif key == ord('x'):
                    print("Exiting data collection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    # Before returning, save whatever data was collected so far
                    if not all_features_df.empty:
                        all_features_df.to_csv(FEATURES_FILE, index=False)
                        print(f"\nPartially collected features saved to {FEATURES_FILE}")
                    else:
                        print("\nNo features collected in this session. Features file not modified.")
                    return

    cap.release()
    cv2.destroyAllWindows()

    # Final save of the combined DataFrame
    if not all_features_df.empty:
        all_features_df.to_csv(FEATURES_FILE, index=False)
        print(f"\nAll collected and existing features combined and saved to {FEATURES_FILE}")
    else:
        print("\nNo features collected or existing. Features file not created/modified.")


# --- Training Function ---
def train_model():
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: Features file not found at {FEATURES_FILE}. Please collect data first.")
        return

    df = pd.read_csv(FEATURES_FILE)
    
    # Drop rows with any NaN values, which can occur if feature extraction failed for some frames
    df.dropna(inplace=True)
    if df.empty:
        print("Error: No valid data in features.csv after dropping NaNs. Cannot train model.")
        return

    X = df.drop('sign', axis=1)
    y = df['sign']

    # Handle cases where there's only one class or too few samples for split
    if len(y.unique()) < 2:
        print(f"Error: Only {len(y.unique())} sign class(es) found ({y.unique()}). Need at least two different signs to train.")
        print("Please collect data for at least two distinct signs.")
        return
    
    # Check for enough samples per class for stratification
    min_samples_per_class = y.value_counts().min()
    # Scikit-learn's train_test_split with stratify needs at least 2 samples for the smallest class in *both* train and test set.
    # So for test_size=0.2, you generally need at least 2 for split to maintain ratio
    if min_samples_per_class < 2:
        print(f"WARNING: Some sign classes have very few samples (min: {min_samples_per_class}). This might cause issues with stratified splitting.")
        print("Proceeding without stratification or with adjusted test_size if necessary.")
        # If any class has only 1 sample, stratification will fail.
        # So, we check if all classes have at least 2 samples to use stratification.
        if all(count >= 2 for count in y.value_counts()):
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            # Fallback for very small classes (e.g., min_samples_per_class == 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print("Note: Stratification skipped due to very small class sizes.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use GridSearchCV for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
    
    print("\n--- Starting SVM training with GridSearchCV ---")
    grid_search.fit(X_train_scaled, y_train)

    model = grid_search.best_estimator_
    print(f"\nBest SVM parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.2f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

# --- Main execution block ---
if __name__ == "__main__":
    while True:
        choice = input("\nChoose an option:\n1. Collect New Data\n2. Train Model\n3. Exit\nEnter choice (1/2/3): ").strip()

        if choice == '1':
            signs = input("Enter signs to collect (comma-separated, e.g., A,B,C). Existing signs will be updated if listed: ").strip().split(',')
            signs = [s.strip().upper() for s in signs if s.strip()] # Clean and uppercase
            if signs:
                collect_data(signs)
            else:
                print("No signs entered. Please enter at least one sign.")
        elif choice == '2':
            train_model()
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")