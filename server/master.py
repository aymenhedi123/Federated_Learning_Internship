from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from threading import Lock
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from preprocessing import drop_null, drop_duplicated, drop_useless_f, x_y_split
app = Flask(__name__)
lock = Lock()
clients_weights = []
clients_losses = []
clients_updates = 0  # Track the number of clients that have sent updates
total_clients = 2  # Total number of clients expected to send updates

# Initialize the best model metrics
best_loss = float('inf')
best_accuracy = 0.0

# Define and compile the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(24,)),  # Adjust input_shape as per your features
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the pre-trained weights
if os.path.exists('global_weights.h5'):
    model.load_weights('global_weights.h5')
    print("Loaded initial model weights from global_weights.h5")
else:
    print("No initial weights found, starting with random weights")

@app.route('/get_model', methods=['GET'])
def get_model():
    weights = model.get_weights()
    print("Global model First weight matrix:\n", weights[0])
    weights_serializable = [w.tolist() for w in weights]  # Convert weights to list for serialization
    return jsonify({'weights': weights_serializable})

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global clients_updates
    print("Updated model weights received from clients")
    data = request.get_json()
    local_weights = [np.array(w) for w in data['weights']]
    local_loss = data['loss']
    local_accuracy = data['accuracy']

    with lock:
        clients_weights.append(local_weights)
        clients_losses.append(local_loss)
        clients_updates += 1

    print("Client weights and loss received")

    # Check if all clients have sent their updates for this iteration
    if clients_updates == total_clients:
        federated_averaging()
        clients_updates = 0  # Reset the counter for the next iteration

    return jsonify({'status': 'success', 'message': 'Client weights and loss received'})

@app.route('/evaluate_global_model', methods=['POST'])
def evaluate_global_model():
    global best_loss, best_accuracy

    # Load the test dataset
    df = pd.read_csv('smoking.csv')

    # Preprocess the dataset
    if df.isnull().sum().any():
        df = drop_null(df)

    if df.duplicated().sum() > 0:
        df = drop_duplicated(df)

    df = drop_useless_f(df)
    X,y = x_y_split(df)
    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)

    # Evaluate the model
    loss, accuracy = model.evaluate(X, y, verbose=0)

    print(f'Evaluation results - Loss: {loss}, Accuracy: {accuracy}')

    # Check if this is the best model so far based on loss or accuracy
    if loss < best_loss or accuracy > best_accuracy:
        if loss < best_loss:
            best_loss = loss
            print(f"New best model found based on loss! New loss: {loss}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best model found based on accuracy! New accuracy: {accuracy}")
        save_best_model()
    else:
        print(f"No improvement in model. Best loss: {best_loss}, Current loss: {loss}, Best accuracy: {best_accuracy}, Current accuracy: {accuracy}")

    return jsonify({'loss': loss, 'accuracy': accuracy})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

def federated_averaging():
    global clients_weights, clients_losses

    with lock:
        if clients_weights:
            # Average each layer's weights across all clients
            new_weights = [np.mean([client_weights[layer] for client_weights in clients_weights], axis=0) for layer in range(len(clients_weights[0]))]
            model.set_weights(new_weights)
            clients_weights = []  # Clear the weights for next iteration
            clients_losses = []   # Clear the losses for next iteration
            print("Global model updated with federated averaging")

            # Evaluate the global model
            evaluate_global_model()
            print("Global model evaluated after federated averaging")

        # Early stopping check
        if all(loss < 0.35 for loss in clients_losses):
            print("Early stopping criteria met. Stopping federated learning.")

def save_best_model():
    # Save the best model weights in a pickle file
    weights = model.get_weights()
    with open('/app/data/best_model.pkl', 'wb') as f:
        pickle.dump(weights, f)
    print("Best model saved with loss:", best_loss, "and accuracy:", best_accuracy)

if __name__ == "__main__":
    import threading
    threading.Thread(target=federated_averaging, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
