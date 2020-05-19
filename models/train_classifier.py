"""
MODEL: TRAIN CLASSIFIER
Udacity - Data Science Nanodegree: Disaster Response Pipeline Project
Script Execution: >  python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Arguments:
    1. Path to SQLite destination database (DisasterResponse.db)
    2. Path to pickle file name where ML model needs to be saved (classifier.pkl)
"""
import sys
import re
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import pickle

def load_data(database_filepath):
    """
    Load Data Function 
    Arguments:
        database_filepath -> path to DisasterResponse.db
    Output:
        X: messages in dataframe
        Y: 36 message classifications
        category_names: used for data visualization.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessages',con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns   
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize Function 
    Arguments:
        text: list of messages 
    Output:
        clean_tokens: tokenized text for ML modeling
    """
    # Replace all urls with a urlplaceholder string, extract them and replace urls with url placeholder string.
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Extract the word tokens from the provided text and use lemmanitizer method.
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # List of clean tokens in messages.
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

#custom estimator class is created with lessons learned in Case Study:Grid Search of Udacity
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class extracting verb of a sentence and it is used ML model.
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def build_model():
    # new model with custom estimator
    pipeline_new = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),            
            
            ('starting_verb', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {   
        'features__text_pipeline__count_vectorizer__ngram_range': ((1, 1), (1, 2)),
        'classifier__estimator__learning_rate': [0.02, 0.04],
        'classifier__estimator__n_estimators': [20, 40] }

    model = GridSearchCV(pipeline_new, param_grid=parameters)
    return model

# test score function
def test_score(y_test, y_pred):
    scores = pd.DataFrame(columns=['Category', 'F_score', 'Precision', 'Recall'])
    num = 0
    for cat in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,num], average='weighted')
        scores.set_value(num+1, 'Category', cat)
        scores.set_value(num+1, 'F_score', f_score)
        scores.set_value(num+1, 'Precision', precision)
        scores.set_value(num+1, 'Recall', recall)
        num += 1
    print('Aggregated F_score:', scores['F_score'].mean())
    print('Aggregated Precision:', scores['Precision'].mean())
    print('Aggregated Recall:', scores['Recall'].mean())
    return scores

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate Model Function
    Argumets:
    1. model: model evaluated
    2. X_test: messages as X values
    3. Y_test: 36 classfications as Y values
    4. category_names: list of category strings
    Output: None / printing f_score, precision and recall for each category.
    """
    #testing scores of new model
    Y_pred = model.predict(X_test)
    final_test_scores = test_score(Y_test, Y_pred)
    final_test_scores   
    pass

def save_model(model, model_filepath):
    """
    Save Model Function: Saving trained model as pickle file.
    Arguments:
        1.model: GridSearchCV
        2.model_filepath: destination path for .pkl file   
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass

def main():
    """
    Train Classifier Main Function
    1. load_data fucntion to extract data from database.
    2. build_model function to create model, then train ML model.
    3. evaluate_model function to observe performance of model.
    4. save_model function to save model as pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()