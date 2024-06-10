import os
import glob
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_audio_files(base_path, max_length=None):
    features = []
    labels = []
    accents = ['Hebron', 'Jerusalem', 'Nablus', 'Ramallah_Reef']

    for accent in accents:
        folder_path = os.path.join(base_path, accent)
        for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # we take 20 mfccs
            if max_length:
                mfccs = pad_sequences(mfccs, maxlen=max_length, padding='post', truncating='post')
            # Reshape MFCCs to add channel dimension
            mfccs = np.expand_dims(mfccs, axis=-1)
            features.append(mfccs)
            labels.append(accent)

    return np.array(features), np.array(labels)


train_base_path = 'training'
test_base_path = 'testing'

### Load and preprocess the data
X_train, y_train = load_audio_files(train_base_path)
X_test, y_test = load_audio_files(test_base_path)

### Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

### Convert labels to categorical (one-hot encoding)
num_classes = len(label_encoder.classes_)
y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)

### Shuffle the data
X_train, y_train_categorical = shuffle(X_train, y_train_categorical, random_state=42)

### Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

### Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

### Train the model
model.fit(X_train, y_train_categorical, batch_size=32, epochs=20, validation_split=0.1)

### Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

### Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

### Print classification report and confusion matrix
print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))
conf_mat = confusion_matrix(y_test_encoded, y_pred_classes)
print("Confusion Matrix:")
print(conf_mat)
