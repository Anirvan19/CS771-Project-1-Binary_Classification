import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------
# Emoticons Dataset - LSTM Preprocessing and KNN Classifier
# -----------------------------------------------------------
# Load datasets
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
test_emoticon_df = pd.read_csv("datasets/test/test_emoticon.csv")

# Prepare data
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
valid_emoticon_Y = valid_emoticon_df['label'].tolist()
test_emoticon_X = test_emoticon_df['input_emoticon'].tolist()

# Encode labels
le = LabelEncoder()
train_emoticon_Y_encoded = le.fit_transform(train_emoticon_Y)
valid_emoticon_Y_encoded = le.transform(valid_emoticon_Y)

# Tokenization and Padding
tokenizer = Tokenizer(char_level=True)  # Character-level tokenization
tokenizer.fit_on_texts(train_emoticon_X)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_emoticon_X)
valid_sequences = tokenizer.texts_to_sequences(valid_emoticon_X)
test_sequences = tokenizer.texts_to_sequences(test_emoticon_X)

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in train_sequences)  # Use the longest sequence length
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Build Neural Network Model for Feature Extraction
embedding_dim = 8  # Dimension for embedding layer
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(16, return_sequences=False))  # LSTM layer for sequence processing
model.add(Dense(8, activation='relu'))  # Dense layer for feature representation
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network for feature extraction
model.fit(train_padded, train_emoticon_Y_encoded, epochs=10, batch_size=32, validation_data=(valid_padded, valid_emoticon_Y_encoded))

# Extract features
train_features = model.predict(train_padded)
valid_features = model.predict(valid_padded)
test_features = model.predict(test_padded)

# Build and train the KNN model using full training data
knn = KNeighborsClassifier()
knn.fit(train_features, train_emoticon_Y_encoded)

# Predict on validation data and calculate accuracy
valid_predictions = knn.predict(valid_features)
valid_accuracy = accuracy_score(valid_emoticon_Y_encoded, valid_predictions)
print(f"Validation Accuracy with KNN: {valid_accuracy:.4f}")

# Predict on the test data
test_predictions = knn.predict(test_features)

# Create the output DataFrame with binary predictions
output_df = pd.DataFrame({
    'predicted_label': test_predictions
})

# Save predictions to a text file with no header and only one column of binary values
output_df.to_csv("pred_emoticon.txt", sep='\t', index=False, header=False)


# -------------------------------------------------------------
# Deep Features Dataset - LSTM Preprocessing and KNN Classifier
# -------------------------------------------------------------



# Preprocess with LSTM (No training, just feature extraction)
def lstm_preprocess(X_train, X_test, lstm_units=64):
    # LSTM model for feature extraction
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Use the LSTM to transform the data (feature extraction)
    X_train_lstm = model.predict(X_train)
    X_test_lstm = model.predict(X_test)

    return X_train_lstm, X_test_lstm

# Load training and validation data
data_train = np.load(r"datasets/train/train_feature.npz")
feat_train = data_train['features']
label_train = data_train['label']

data_valid = np.load(r"datasets/valid/valid_feature.npz")
feat_valid = data_valid['features']
label_valid = data_valid['label']

# Load test data
data_test = np.load(r"datasets/test/test_feature.npz")
feat_test = data_test['features']

# Assuming your data is 3D (samples, time_steps, features)
X_train = feat_train
X_val = feat_valid
X_test = feat_test

# Apply LSTM for feature extraction
X_train_lstm, X_val_lstm = lstm_preprocess(X_train, X_val)
_, X_test_lstm = lstm_preprocess(X_train, X_test)

y_train = label_train
y_val = label_valid

# Define the KNN model and hyperparameter grid for tuning
knn_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],        # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function
    'metric': ['euclidean', 'manhattan'] # Distance metric
}

# Perform grid search to find the best hyperparameters using the validation set
grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_lstm, y_train)

# Get the best KNN model
best_knn_model = grid_search.best_estimator_

# Train the KNN model with the best hyperparameters on the full training data
best_knn_model.fit(X_train_lstm, y_train)

# Predict on the validation set and test set
y_val_pred = best_knn_model.predict(X_val_lstm)
y_test_pred = best_knn_model.predict(X_test_lstm)

# Calculate and print validation accuracy for tuning
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy with best hyperparameters: {val_accuracy:.4f}")

# Save predictions on the test set to pred_deepfeat.txt
with open("pred_deepfeat.txt", "w") as f:
    for pred in y_test_pred:
        f.write(f"{int(pred)}\n")

# ----------------------------------------------
# Text Sequence Dataset - SVM Classifier
# ----------------------------------------------

train_data = pd.read_csv("datasets/train/train_text_seq.csv")
test_data = pd.read_csv("datasets/test/test_text_seq.csv")

X_train = train_data['input_str'].values  
Y_train = train_data['label'].values  

X_train_cleaned = train_data['input_str'].str.extractall('(\d)').unstack().astype(int).values.tolist()
X_test_cleaned = test_data['input_str'].str.extractall('(\d)').unstack().astype(int).values.tolist()

max_length = max(max(len(seq) for seq in X_train_cleaned), max(len(seq) for seq in X_test_cleaned))
X_train_padded = pad_sequences(X_train_cleaned, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_cleaned, maxlen=max_length, padding='post')

encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
X_train_onehot = encoder.fit_transform(X_train_padded).toarray()
X_test_onehot = encoder.transform(X_test_padded).toarray()  

svm_model = SVC(C=1/0.01)
svm_model.fit(X_train_onehot, Y_train)
Y_test_pred = svm_model.predict(X_test_onehot)

with open("pred_textseq.txt", "w") as f:
    for label in Y_test_pred:
        f.write(f"{label}\n")
# -----------------------------------------------
# Combined Features Dataset - Random Forest
# -----------------------------------------------

# Load emoticon datasets again
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
test_emoticon_df = pd.read_csv("datasets/test/test_emoticon.csv")

# Load text sequence datasets again
train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
valid_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
test_seq_df = pd.read_csv("datasets/test/test_text_seq.csv")

# Load deep features dataset again
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
test_feat = np.load("datasets/test/test_feature.npz", allow_pickle=True)

# Prepare data for emoticons
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
valid_emoticon_Y = valid_emoticon_df['label'].tolist()
test_emoticon_X = test_emoticon_df['input_emoticon'].tolist()

# Prepare data for text sequences
train_seq_X = train_seq_df['input_str'].tolist()
valid_seq_X = valid_seq_df['input_str'].tolist()
test_seq_X = test_seq_df['input_str'].tolist()

# Prepare data for deep features
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']
test_feat_X = test_feat['features']

# Label encoding (for labels consistency)
le = LabelEncoder()
train_emoticon_Y_encoded = le.fit_transform(train_emoticon_Y)
valid_emoticon_Y_encoded = le.transform(valid_emoticon_Y)

# Step 1: Process emoticons (character-level tokenization and padding)
tokenizer_emoticon = Tokenizer(char_level=True)
tokenizer_emoticon.fit_on_texts(train_emoticon_X)

train_emoticon_seq = tokenizer_emoticon.texts_to_sequences(train_emoticon_X)
valid_emoticon_seq = tokenizer_emoticon.texts_to_sequences(valid_emoticon_X)
test_emoticon_seq = tokenizer_emoticon.texts_to_sequences(test_emoticon_X)

max_len_emoticon = max(len(seq) for seq in train_emoticon_seq)

train_emoticon_padded = pad_sequences(train_emoticon_seq, maxlen=max_len_emoticon, padding='post')
valid_emoticon_padded = pad_sequences(valid_emoticon_seq, maxlen=max_len_emoticon, padding='post')
test_emoticon_padded = pad_sequences(test_emoticon_seq, maxlen=max_len_emoticon, padding='post')

# Step 2: Process text sequences (word-level tokenization and padding)
tokenizer_seq = Tokenizer()
tokenizer_seq.fit_on_texts(train_seq_X)

train_seq_seq = tokenizer_seq.texts_to_sequences(train_seq_X)
valid_seq_seq = tokenizer_seq.texts_to_sequences(valid_seq_X)
test_seq_seq = tokenizer_seq.texts_to_sequences(test_seq_X)

max_len_seq = max(len(seq) for seq in train_seq_seq)

train_seq_padded = pad_sequences(train_seq_seq, maxlen=max_len_seq, padding='post')
valid_seq_padded = pad_sequences(valid_seq_seq, maxlen=max_len_seq, padding='post')
test_seq_padded = pad_sequences(test_seq_seq, maxlen=max_len_seq, padding='post')

# Step 3: Flatten deep features (reshape 3D arrays to 2D)
train_feat_X_flat = train_feat_X.reshape(train_feat_X.shape[0], -1)
valid_feat_X_flat = valid_feat_X.reshape(valid_feat_X.shape[0], -1)
test_feat_X_flat = test_feat_X.reshape(test_feat_X.shape[0], -1)

# Step 4: Combine the features
train_combined_X = np.hstack([train_feat_X_flat, train_emoticon_padded, train_seq_padded])
valid_combined_X = np.hstack([valid_feat_X_flat, valid_emoticon_padded, valid_seq_padded])
test_combined_X = np.hstack([test_feat_X_flat, test_emoticon_padded, test_seq_padded])

# Step 5: Scale the features
scaler = StandardScaler()
train_combined_X = scaler.fit_transform(train_combined_X)
valid_combined_X = scaler.transform(valid_combined_X)
test_combined_X = scaler.transform(test_combined_X)

# Train a Random Forest classifier on combined features
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(train_combined_X, train_emoticon_Y_encoded)

# Validation accuracy for Random Forest
valid_pred_combined = random_forest_model.predict(valid_combined_X)
valid_accuracy_combined = accuracy_score(valid_emoticon_Y_encoded, valid_pred_combined)
print(f"Validation Accuracy with Random Forest (Combined Features): {valid_accuracy_combined:.4f}")

# Test predictions for combined features
test_pred_combined = random_forest_model.predict(test_combined_X)

# Save test predictions for combined features
with open("pred_combined.txt", "w") as f:
    for label in test_pred_combined:
        f.write(f"{label}\n")
