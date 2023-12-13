# Machine Learning Classification and CNN for Image Classification

## Group Members
1. Aaryan Dhore (knc8xp)
2. Dhriti Gampa (jwr9ew)

## File 1: Machine Learning Classification

### Introduction

This Python script is designed for machine learning classification tasks using various classifiers. It includes functions for data preprocessing, cross-validation, and model evaluation. The classifiers implemented in this script include Logistic Regression, K Nearest Neighbors, Decision Trees, Support Vector Machines, Random Forests, and AdaBoost.

### Dependencies

The script requires the following Python libraries:

- pandas
- scikit-learn
- numpy

### Functions

1. **`preprocess_data(dataset)`**

   This function takes a dataset as input and performs the following steps:
   - Separates features (X) and classifications (y).
   - Splits the data into training and testing sets (80-20 split).
   - Encodes categorical features using LabelEncoder.
   - Normalizes continuous features using StandardScaler.

2. **`evaluate_classifier(clf, X_train, X_test, y_train, y_test, cv=10)`**

   This function evaluates a classifier using 10-fold cross-validation and provides metrics such as accuracy, precision, recall, F1 score, and AUC.

3. **Classifier Functions**
   - `logistic_regression(X_train, y_train)`
   - `k_nearest_neighbor(X_train, y_train)`
   - `decision_tree(X_train, y_train)`
   - `support_vector_machine(X_train, y_train)`
   - `random_forest(X_train, y_train, max_depth=None)`
   - `boosting(X_train, y_train)`

   These functions train the specified classifiers on the provided training data.

4. **Training and Testing Model Metrics on Datasets**

   The script loads two datasets (`project3_dataset1.txt` and `project3_dataset2.txt`), preprocesses them, and trains multiple classifiers on each dataset. Model performance metrics are displayed for both training and testing sets.

### Execution

To run the script, ensure that the required datasets are available and adjust the file paths accordingly. Execute the script, and the results will be printed, including the best hyperparameters for the MLP classifier.

## File 2: Convolutional Neural Network (CNN) for Image Classification

### Introduction

This Python script implements a Convolutional Neural Network (CNN) for image classification using the MNIST dataset. It utilizes TensorFlow and Keras for building and training the CNN. It then goes over several different models comparing their performance with different hyperparameters. 

### Dependencies

The script requires the following Python libraries:

- mnist_loader
- numpy
- tensorflow
- matplotlib
- scikit-learn

### Functions

1. **Loading and Preprocessing Data**

   The script uses the `mnist_loader` module to load and preprocess the MNIST dataset.

2. **Building the CNN Model**

   The CNN model is defined using the Keras Sequential API, consisting of convolutional layers, max-pooling layers, and dense layers.

3. **Compiling and Training the Model**

   The model is compiled using the Adam optimizer and categorical crossentropy loss. Training is performed for a specified number of epochs.

4. **Model Evaluation**

   The script evaluates the trained model on a validation set and displays the training and validation loss over epochs.

5. **Metrics and Visualization**

   It calculates and prints the accuracy and classification report for the test set. Additionally, it visualizes a subset of incorrectly classified images.

6. **Hyperparameter Tuning**

   Multiple different models are made and their performances are compared to each other with different hyperparameters. Such aspects that are changed is the network architecture (shallow v deep), batch size, and number of epochs. 


### Execution

Ensure that the required libraries are installed before running the script. Execute the script to load, preprocess, train, and evaluate the CNN model on the MNIST dataset. The results and visualizations will be displayed accordingly. Adjust hyperparameters or the number of epochs as needed.
