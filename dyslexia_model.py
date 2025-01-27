import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import joblib

class DyslexiaDetector:

    def __init__(self):
        try:
            self.model = joblib.load('dyslexia_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        except:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

            self.scaler = StandardScaler()

    def extract_features(self, fixation_data):

        """Extract relevant features from fixation data"""

        features = {

            'mean_fixation_duration': np.mean(fixation_data['fixation_duration']),

            'std_fixation_duration': np.std(fixation_data['fixation_duration']),

            'total_fixations': len(fixation_data),

            'mean_saccade_length': np.mean(np.sqrt(np.diff(fixation_data['fixation_x'])**2 + 

                                                  np.diff(fixation_data['fixation_y'])**2)),

            'mean_x_position': np.mean(fixation_data['fixation_x']),

            'mean_y_position': np.mean(fixation_data['fixation_y'])

        }

        return pd.Series(features)

    def train(self, data_path):

        """Train the model using the ETDD70 dataset"""

        # Load and preprocess the dataset

        df = pd.read_csv(data_path)

        # Extract features for each participant

        features_list = []

        labels = []

        for participant in df['participant_id'].unique():

            participant_data = df[df['participant_id'] == participant]

            features = self.extract_features(participant_data)

            features_list.append(features)

            labels.append(participant_data['has_dyslexia'].iloc[0])

        X = pd.DataFrame(features_list)

        y = np.array(labels)

        # Split and scale the data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train the model

        self.model.fit(X_train_scaled, y_train)

        # Save the model and scaler

        joblib.dump(self.model, 'dyslexia_model.pkl')

        joblib.dump(self.scaler, 'scaler.pkl')

    def predict(self, fixation_data):

        """Predict whether a person has dyslexia based on their fixation data"""

        features = self.extract_features(fixation_data)

        features_scaled = self.scaler.transform(features.values.reshape(1, -1))

        prediction = self.model.predict(features_scaled)

        probability = self.model.predict_proba(features_scaled)

        return prediction[0], probability[0]
