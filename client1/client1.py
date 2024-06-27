import requests
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import sleep
from preprocessing import drop_null, drop_duplicated, drop_useless_f, x_y_split

model = Sequential([
    Dense(64, activation='relu', input_shape=(24,)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fetching server_model weights with retries
def get_model_with_retries(url, max_retries=5, backoff=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                sleep(backoff)
            else:
                raise

# Sending updated model weights to the master
def send_updated_model(update_url_weights, updated_weights, loss, accuracy):
    weights_serializable = [w.tolist() for w in updated_weights]
    response = requests.post(update_url_weights, json={'weights': weights_serializable, 'loss': loss, 'accuracy': accuracy})
    response.raise_for_status()
    print(f"Updated model sent to {update_url_weights}, response: {response.json()}")

# Preprocessing and evaluation
def main():
    print("Reading data")
    df = pd.read_csv('smoking_subset_1.csv')
    print(df.head())
    if df.isnull().sum().any():
        df = drop_null(df)

    if df.duplicated().sum() > 0:
        df = drop_duplicated(df)

    df = drop_useless_f(df)
    X,y = x_y_split(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    url = 'http://master:5000/get_model'
    update_url_weights = 'http://master:5000/update_weights'
    max_iterations = 20
    for iteration in range(max_iterations):
        response = get_model_with_retries(url)
        weights = response.json()['weights']
        weights = [tf.convert_to_tensor(w) for w in weights]
        model.set_weights(weights)
        print(f"Model weights set in client from the master server, iteration {iteration + 1}")

        best_accuracy = 0.0
        best_weights = None

        for epoch in range(2):
            history = model.fit(X_train, y_train, epochs=1, verbose=1)
            print(f"Epoch {epoch + 1}/{2}")

            if history.history['accuracy'][0] > best_accuracy:
                best_accuracy = history.history['accuracy'][0]
                best_weights = model.get_weights()

        print("Local training completed")

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

        if best_weights is not None:
            model.set_weights(best_weights)
            print(f"Best accuracy achieved during training: {best_accuracy}")

        updated_weights = model.get_weights()

        
        print(f"Client {iteration + 1} updated weights (first layer first neuron): {updated_weights[0][0]}")

        send_updated_model(update_url_weights, updated_weights, loss, accuracy)

        if loss < 0.2:
            print("Early stopping criteria met. Stopping local training.")
            break

    print("Client1 training completed.")

if __name__ == "__main__":
    main()
