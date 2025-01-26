import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

class DyslexiaDataProcessor:

    def __init__(self):

        self.scaler = StandardScaler()

    def extract_features(self, fixation_data):

        """Extract features from fixation data"""

        features = {

            'mean_fixation_duration': np.mean(fixation_data['duration']),

            'std_fixation_duration': np.std(fixation_data['duration']),

            'total_fixations': len(fixation_data),

            'mean_saccade_length': np.mean(np.diff(fixation_data[['x', 'y']].values, axis=0)),

            'std_saccade_length': np.std(np.diff(fixation_data[['x', 'y']].values, axis=0))

        }

        return pd.Series(features)

    def prepare_data(self, data_path):

        """Prepare data for training"""

        # Load ETDD70 dataset

        df = pd.read_csv(data_path)

        # Group by participant and extract features

        features = df.groupby('participant_id').apply(self.extract_features)

        # Scale features

        X = self.scaler.fit_transform(features)

        y = df.groupby('participant_id')['has_dyslexia'].first().values

        return train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

class DyslexiaDetector:

    def __init__(self):

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.processor = DyslexiaDataProcessor()

    def train(self, data_path):

        """Train the model"""

        X_train, X_test, y_train, y_test = self.processor.prepare_data(data_path)

        self.model.fit(X_train, y_train)

        # Evaluate model

        y_pred = self.model.predict(X_test)

        print("Model Accuracy:", accuracy_score(y_test, y_pred))

        print("\nClassification Report:")

        print(classification_report(y_test, y_pred))

    def predict(self, fixation_data):

        """Predict if a person has dyslexia based on their fixation data"""

        features = self.processor.extract_features(fixation_data)

        features_scaled = self.processor.scaler.transform(features.values.reshape(1, -1))

        return self.model.predict_proba(features_scaled)[0]
