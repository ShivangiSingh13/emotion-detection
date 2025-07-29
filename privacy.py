import os
import shutil

# Define the directory where session data (like diary entries) is stored
SESSION_DATA_DIR = 'session_data'

def clear_session_data():
    """
    Clears all recorded session data, including diary entries.
    """
    if os.path.exists(SESSION_DATA_DIR):
        try:
            shutil.rmtree(SESSION_DATA_DIR)
            print(f"INFO: All session data in '{SESSION_DATA_DIR}' cleared.")
            # Optionally re-create the directory if needed immediately
            os.makedirs(SESSION_DATA_DIR, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Failed to clear session data: {e}")
            # Consider raising the exception or showing a user-friendly error message in the GUI
    else:
        print(f"INFO: Session data directory '{SESSION_DATA_DIR}' does not exist. Nothing to clear.")

if __name__ == '__main__':
    # Example usage:
    print("Testing clear_session_data...")
    
    # Create a dummy file for testing
    os.makedirs(SESSION_DATA_DIR, exist_ok=True)
    with open(os.path.join(SESSION_DATA_DIR, 'test.txt'), 'w') as f:
        f.write("This is a test file.")
    print(f"Created dummy file: {os.path.join(SESSION_DATA_DIR, 'test.txt')}")

    clear_session_data()

    if not os.path.exists(os.path.join(SESSION_DATA_DIR, 'test.txt')) and os.path.exists(SESSION_DATA_DIR):
        print("Test successful: Dummy file cleared and directory re-created.")
    elif not os.path.exists(SESSION_DATA_DIR):
        print("Test partially successful: Directory was removed. Consider if it should be re-created by default.")
    else:
        print("Test failed: Dummy file still exists.")