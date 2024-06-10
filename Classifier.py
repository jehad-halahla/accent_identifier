
#inckude the necessary libraries if they are not included
import os
import glob
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# method to load audio files
def load_audio_files(base_path):
    features = []
    labels = []
    accents = ['Hebron', 'Jerusalem', 'Nablus','Ramallah_Reef']

    for accent in accents:
        folder_path = os.path.join(base_path, accent)
        for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)#we take 20 mfccs
            mfccs_mean = np.mean(mfccs, axis=1)
            features.append(mfccs_mean)
            labels.append(accent)

    return np.array(features), np.array(labels)

class classifier:
    '''this class is deticated to read training and testing data from testing and training files,
      and train the SVM model, along with scaling the features, and training the KNN model using grid search'''
    
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.features_train, self.labels_train = load_audio_files(self.train_dir)#training features and labels
        self.features_test, self.labels_test = load_audio_files(self.test_dir)# testing features and labels
        #best models choosen by grid search
        self.best_svm = None
        self.best_knn = None
        self.kmeans = None
    

    def scale_values(self):
        '''this method is used to scale the features for both train and test data using StandardScaler'''
        scaler = StandardScaler()
        self.features_train = scaler.fit_transform(self.features_train)
        self.features_test = scaler.transform(self.features_test)
    
    def choose_best_SVM(self):
        '''this method is used to train the SVM models and choose the best model using grid search'''
       
        # Use Grid Search to find the best parameters for SVM
        parameter_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'class_weight': ['balanced']}
        grid_search = GridSearchCV(SVC(), parameter_grid, cv=5)
        grid_search.fit(self.features_train, self.labels_train)
        # Get the best model from Grid Search
        self.best_svm = grid_search.best_estimator_

    def perform_clustering(self):
        '''This method performs K-Means clustering on the training data'''
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(self.features_train)
        # Evaluate clustering
        silhouette_avg = silhouette_score(self.features_train, cluster_labels)
        label_mapping = {label: idx for idx, label in enumerate(np.unique(self.labels_train))}
        numerical_labels = np.array([label_mapping[label] for label in self.labels_train])

    
    def choose_best_KNN(self):
            '''This method is used to train the KNN models and choose the best model using grid search'''
            # Use Grid Search to find the best parameters for KNN
            parameter_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree']}
            grid_search = GridSearchCV(KNeighborsClassifier(), parameter_grid, cv=5)
            grid_search.fit(self.features_train, self.labels_train)
            # Get the best model from Grid Search
            self.best_knn = grid_search.best_estimator_

    def evaluate_SVM(self):
        """This method evaluates the best SVM , print the accuracy, confusion matrix and classification report"""
        y_pred = self.best_svm.predict(self.features_test)
        accuracy = accuracy_score(self.labels_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        # Print confusion matrix and classification report
        #cm = confusion_matrix(self.labels_test, y_pred, labels=self.best_svm.classes_)
        #sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.best_svm.classes_, yticklabels=self.best_svm.classes_,cmap='binary', linewidths=1)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # plt.show()
        #print(classification_report(self.labels_test, y_pred, target_names=self.best_svm.classes_))




if __name__ == '__main__':
    print('Reading training and testing data...')
    train_base_path = 'training'
    test_base_path = 'testing'
    classifier_obj = classifier(train_base_path, test_base_path)
    classifier_obj.scale_values()
    classifier_obj.choose_best_SVM()
    classifier_obj.evaluate_SVM()
