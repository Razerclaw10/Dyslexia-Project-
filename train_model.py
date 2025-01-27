# train_model.py

#import pandas as pd

from dyslexia_model import DyslexiaDetector

# Function to create and save the dataset (if not already available)

"""def create_dataset():

    data = {

        'participant_id': ['1', '1', '1', '2', '2'],  # Example participant IDs

        'fixation_duration': [150, 200, 180, 120, 130],  # Example fixation durations

        'fixation_x': [200, 250, 220, 150, 180],  # Example x-coordinates

        'fixation_y': [300, 350, 280, 240, 260],  # Example y-coordinates

        'is_fixation': [True, True, True, True, True],  # Example fixation flags

        'has_dyslexia': [1, 1, 1, 0, 0]  # Example dyslexia labels for training

    }

    # Create DataFrame

    df = pd.DataFrame(data)

    # Save to CSV file

    df.to_csv('path_to_ETDD70_dataset.csv', index=False)

    print("Dataset created and saved as path_to_ETDD70_dataset.csv")"""

# Main training function

def main():

    # Create dataset if it doesn't exist

  #  create_dataset()

    # Train the model

    detector = DyslexiaDetector()

    detector.train('path_to_ETDD70_dataset.csv')

if __name__ == "__main__":

    main()
