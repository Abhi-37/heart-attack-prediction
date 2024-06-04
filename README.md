# heart-attack-prediction
This script performs KMeans clustering and trains a neural network to predict heart disease, including data preprocessing, clustering visualization, and model evaluation.

**Loading and Preprocessing Data:**
- Loading Data: We start by loading the heart disease dataset using Pandas.
- Exploratory Data Analysis (EDA): Initial visualization includes a count plot of the target variable (HeartDisease) and a correlation heatmap of the features to understand their relationships.
- Encoding Categorical Variables: We use LabelEncoder to transform categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, and ST_Slope) into numerical values.
- Handling Missing Values: Any missing values are filled with the mean of the respective columns to ensure completeness of the dataset.

**KMeans Clustering:**
- Standardizing Features: We standardize the features using StandardScaler to ensure all features contribute equally to distance calculations in clustering.
- Applying KMeans: We apply KMeans clustering with n_clusters=2 to group the data into two clusters, possibly representing healthy and diseased individuals.
- Adding Cluster Labels: The resulting cluster labels are appended as an additional feature to the dataset.

**Visualizing Clusters:**
- Principal Component Analysis (PCA): We use PCA to reduce the dimensionality of the dataset to 2 components for easier visualization.
- Cluster Visualization: We plot the data points with cluster labels to visualize the clusters formed by KMeans.
- True Labels Visualization: We also plot the actual labels to compare and see how well the clusters align with the true classes.

**Data Splitting:**
- Train-Test Split: The dataset is split into training and testing sets using train_test_split to evaluate the model's performance on unseen data.

**Neural Network Model:**
- Model Definition: We define a neural network using TensorFlow Keras. The network consists of:
- An input layer with 64 neurons and ReLU activation.
- A dropout layer with a dropout rate of 0.2 to prevent overfitting.
- A hidden layer with 32 neurons and ReLU activation.
- An output layer with a sigmoid activation for binary classification.
- Model Compilation: The model is compiled using binary cross-entropy loss and the Adam optimizer.

**Model Training:**
- Training: We train the neural network on the training data for 100 epochs with a batch size of 10. Validation data is used to monitor the model's performance during training.

**Model Evaluation:**
- Predictions: After training, we evaluate the model on the testing set and make probability predictions.
- Classification Report: We generate a classification report to summarize the precision, recall, and F1-score.
- Confusion Matrix: We compute and display the confusion matrix to visualize the performance in terms of true and false positives/negatives.
- Accuracy and ROC AUC Score: We calculate the accuracy and ROC AUC score to quantify the model's performance.

**Visualizing Evaluation Results:**
- Confusion Matrix Plot: A heatmap of the confusion matrix is plotted to provide a visual representation of prediction accuracy.
- ROC Curve: We plot the ROC curve to illustrate the trade-off between true positive rate and false positive rate at various thresholds.

By following this comprehensive workflow, we leverage both unsupervised learning (KMeans clustering) to uncover hidden patterns and supervised learning (neural network) to make accurate predictions, ultimately improving the prediction of heart disease.
