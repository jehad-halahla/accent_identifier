import os
import glob
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

# Function to extract features
def extract_features(y, sr, window_size=10, step_size=5):
    features = []
    for start in range(0, len(y) - window_size * sr, step_size * sr):
        end = start + window_size * sr
        segment = y[start:end]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_median = np.median(mfccs, axis=1)
        mfccs_percentiles = np.percentile(mfccs, [25, 50, 75], axis=1)
        mfccs_features = np.concatenate([mfccs_mean, mfccs_std, mfccs_median, mfccs_percentiles.flatten()])
        features.append(mfccs_features)
    return np.array(features)


# Function to load audio files
def load_audio_files(base_path):
    features = []
    labels = []
    accents = ['Hebron', 'Jerusalem', 'Nablus', 'Ramallah_Reef']

    for accent in accents:
        folder_path = os.path.join(base_path, accent)
        for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
            y, sr = librosa.load(file_path, sr=None)
            segment_features = extract_features(y, sr)
            features.extend(segment_features)
            labels.append(accent)

    return np.array(features), np.array(labels)

# Load data
train_base_path = 'training'
test_base_path = 'testing'
features_train, labels_train = load_audio_files(train_base_path)
features_test, labels_test = load_audio_files(test_base_path)

# Scale features
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Choose best SVM
parameter_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(SVC(), parameter_grid, cv=5, n_jobs=-1)
grid_search.fit(features_train, labels_train)
best_svm = grid_search.best_estimator_

# Evaluate SVM
y_pred = best_svm.predict(features_test)
accuracy = accuracy_score(labels_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
cm = confusion_matrix(labels_test, y_pred, labels=best_svm.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=best_svm.classes_, yticklabels=best_svm.classes_, cmap='binary', linewidths=1)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(labels_test, y_pred, target_names=best_svm.classes_))
