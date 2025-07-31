Text Classification Project
This project implements a text classification system to identify "bullying" or "non-bullying" text using various machine learning algorithms. The system preprocesses text data using TF-IDF vectorization and then applies trained models for prediction.

Project Structure
The project consists of the following key files:

main.py: The main Python script containing the data loading, preprocessing, model training, evaluation, and prediction logic.

tfidfVectorizer.pkl: A pickled TfidfVectorizer object, likely containing the vocabulary fitted on the training data. This is used to transform new text into numerical features.

LinearSVC.pkl: A pickled LinearSVC (Linear Support Vector Classification) model, trained for text classification.

SGDClassifier.pkl: A pickled SGDClassifier (Stochastic Gradient Descent Classifier) model.

BaggingClassifier.pkl: A pickled BaggingClassifier model, which is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregates their individual predictions (either by voting or by averaging) to form a final prediction.

AdaBoostClassifier.pkl: A pickled AdaBoostClassifier model, a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted so that subsequent classifiers focus more on difficult cases.

DecisionTreeClassifier.pkl: A pickled DecisionTreeClassifier model.

LogisticRegression.pkl: A pickled LogisticRegression model.

MultinomialNB.pkl: A pickled MultinomialNB (Multinomial Naive Bayes) model, suitable for text classification with discrete features (like word counts or TF-IDF values).

Setup and Installation
To run this project, you'll need Python and several libraries.

Clone the repository (if applicable) or ensure all files are in the same directory.

Install the required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk

Download NLTK data:
The script uses NLTK for tokenization. You might need to download the necessary NLTK data:

import nltk
nltk.download('punkt')

Ensure the dataset is available:
The main.py script expects a CSV file named badwords.csv. Please place your dataset in the specified path or update the path in main.py:
df = pd.read_csv(r"C:\Users\saqla\Downloads\badwords.csv")

Usage
The main.py script can be run directly to perform data manipulation, distribution analysis, model training (if perform_training is set to True), evaluation, and a sample prediction.

To run the script:

python main.py

Key Functions in main.py:
perfrom_data_manipulation(): Reads the badwords.csv file and transforms -1 labels to 1.

performdatadistribution(df): Analyzes and visualizes the distribution of labels in the dataset.

perform_data_preprocessing(df): Cleans and preprocesses the text data (e.g., removing URLs, mentions, hashtags, special characters, stopwords, and performing stemming).

perform_data_vectorization(X_train, X_test): Converts text data into numerical TF-IDF features. This function also saves the fitted TfidfVectorizer to tfidfVectorizer.pkl.

perform_training(X_train, Y_train, X_test, Y_test): Trains and evaluates multiple classification models (Logistic Regression, Multinomial Naive Bayes, Decision Tree, Linear SVC, SGD Classifier, AdaBoost, Bagging Classifier, Random Forest, XGBoost). It also saves the trained models as .pkl files.

tuning(clf, param_dict, X_train, Y_train, X_test, Y_test): Performs GridSearchCV for hyperparameter tuning on a given classifier.

Prediction Example: The script includes a section at the end to load a pre-trained LinearSVC.pkl model and tfidfVectorizer.pkl to make a prediction on a sample text:

data = ['You are so beautiful']
tfidf_vector = TfidfVectorizer(stop_words=content_list, lowercase=True, vocabulary=pickle.load(open("tfidfVectorizer.pkl","rb")))
preprocessed_data = tfidf_vector.fit_transform(data)
trained_model = pickle.load(open('LinearSVC.pkl', 'rb'))
print(trained_model.predict(preprocessed_data))
if(trained_model.predict(preprocessed_data)==1):
    print("Bulling")
else:
    print("Non-Bulling")

This part demonstrates how to use the saved models for inference.

Models
The project utilizes and saves the following trained machine learning models:

LinearSVC.pkl

SGDClassifier.pkl

BaggingClassifier.pkl

AdaBoostClassifier.pkl

DecisionTreeClassifier.pkl

LogisticRegression.pkl

MultinomialNB.pkl

These .pkl files are binary files representing the serialized Python objects (the trained models). They can be loaded using pickle.load() for direct use in prediction without retraining.

Data
The dataset expected is badwords.csv. It should contain a column named label which is manipulated to contain 0 (Non-Bullying) and 1 (Bullying) values. The text content for classification is assumed to be in another column, which is processed by the perform_data_preprocessing function.

Visualizations
The main.py script generates plots to visualize data distribution and algorithm performance (Accuracy, F1-Score, Precision, Recall, and Prediction Time). These plots are displayed when the script is run.

Contributing
Feel free to fork this repository and contribute!
