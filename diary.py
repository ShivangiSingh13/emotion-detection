import pandas as pd
import os
from datetime import datetime

# Define the path for the diary CSV file
DIARY_FILE = os.path.join('session_data', 'emotion_diary.csv')

def log_emotion(emotion, reason):
    """
    Logs a detected emotion and its reason to a CSV file.
    """
    # Ensure the session_data directory exists
    os.makedirs(os.path.dirname(DIARY_FILE), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{'timestamp': timestamp, 'emotion': emotion, 'reason': reason}])

    if os.path.exists(DIARY_FILE):
        # Append without writing header if file exists
        new_entry.to_csv(DIARY_FILE, mode='a', header=False, index=False)
    else:
        # Write with header if file doesn't exist
        new_entry.to_csv(DIARY_FILE, mode='w', header=True, index=False)
    # print(f"INFO: Logged emotion: {emotion} - {reason}") # Uncomment for verbose logging

def get_diary_data():
    """
    Retrieves all logged emotion data from the CSV file.
    Returns: pandas DataFrame
    """
    if os.path.exists(DIARY_FILE):
        try:
            df = pd.read_csv(DIARY_FILE)
            return df
        except pd.errors.EmptyDataError:
            print("INFO: Diary file is empty.")
            return pd.DataFrame(columns=['timestamp', 'emotion', 'reason'])
        except Exception as e:
            print(f"ERROR: Failed to read diary data: {e}")
            return pd.DataFrame(columns=['timestamp', 'emotion', 'reason'])
    else:
        return pd.DataFrame(columns=['timestamp', 'emotion', 'reason'])

if __name__ == '__main__':
    # Example Usage:
    print("--- Testing Diary Module ---")
    
    # Clear existing data for a clean test
    from privacy import clear_session_data
    clear_session_data()
    print("Cleared existing diary data for test.")

    # Log some dummy emotions
    log_emotion('happy', 'Just saw a puppy.')
    log_emotion('neutral', 'Working diligently.')
    log_emotion('sad', 'Missed my bus.')
    log_emotion('surprise', 'Found money!')
    log_emotion('neutral', 'Listening to music.')
    
    print("\nLogged dummy emotions. Reading data:")
    df = get_diary_data()
    print(df)
    
    # Verify file existence
    if os.path.exists(DIARY_FILE):
        print(f"\nDiary file exists at: {DIARY_FILE}")
    else:
        print("\nDiary file was NOT created.")

    print("\n--- Diary Module Test Complete ---")