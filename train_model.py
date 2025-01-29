import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import tensorflow as tf

def train_new_model(data_path):

    # Load data

    df = pd.read_csv(data_path)

    # Select features and target

    X = df[['fixation_x', 'fixation_y', 'fixation_duration']]

    y = df['has_dyslexia']

    # Preprocess

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Split data

    X_train, X_test, y_train, y_test = train_test_split(

        X_scaled, y, test_size=0.2, random_state=42

    )

    # Create and train model

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])

    model.compile(optimizer='adam',

                 loss='binary_crossentropy',

                 metrics=['accuracy'])

    model.fit(X_train, y_train,

              validation_data=(X_test, y_test),

              epochs=10,

              batch_size=32)

    # Save model

    model.save('dyslexia_model.h5')

    return model

if __name__ == "__main__":

    train_new_model('path_to_ETDD70_dataset.csv')
