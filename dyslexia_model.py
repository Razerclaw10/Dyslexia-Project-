import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import joblib

class DyslexiaDetector:

    def __init__(self):

        try:

            self.model = tf.keras.models.load_model('dyslexia_model.h5')

            self.scaler = joblib.load('scaler.pkl')

        except:

            self.model = self._build_model()

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
        
    def _build_model(self):

        model = tf.keras.Sequential([

            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),

            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu'),

            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(16, activation='relu'),

            tf.keras.layers.Dense(1, activation='sigmoid')

        ])

        model.compile(

            optimizer='adam',

            loss='binary_crossentropy',

            metrics=['accuracy']

        )

        return model

    def train(self, data_path):

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

        X_train_scaled = self.scaler.fit_transform(X_train)

        X_test_scaled = self.scaler.transform(X_test)

        # Train the model

        self.model.fit(

            X_train_scaled, 

            y_train,

            epochs=100,

            batch_size=32,

            validation_data=(X_test_scaled, y_test),

            verbose=1

        )

        # Save the model and scaler

        self.model.save('dyslexia_model.h5')

        joblib.dump(self.scaler, 'scaler.pkl')

    def predict(self, fixation_data):

        features = self.extract_features(fixation_data)

        features_scaled = self.scaler.transform(features.values.reshape(1, -1))

        probability = self.model.predict(features_scaled)[0][0]

        prediction = (probability > 0.5).astype(int)

        return prediction, np.array([1 - probability, probability])
