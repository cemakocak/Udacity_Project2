"""
PROCESS DATA: CLEAN and UPLOAD DATA into SQLite Database
Udacity - Data Science Nanodegree: Disaster Response Pipeline Project
Script Execution: > python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
Arguments:
    1. CSV file containing messages: disaster_messages.csv
    2. CSV file containing categories: disaster_categories.csv
    3. SQLite database for upload: DisasterResponse.db
"""
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data Function    
    Arguments:
        messages_filepath: path of messages csv file
        categories_filepath: path of categories csv file
    Output:
        df: DataFrame including merged data of messages and categories files according id column. 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 

def clean_data(df):
    """
    Clean Data Function
    Arguments:
        df: raw data in dataframe
    Outputs:
        df: clean data in dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories and apply a lambda function that takes everything. 
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save Data Function 
    Arguments:
        df: clean data in dataframe
        database_filename: SQLite database file destionation
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterMessages', engine, index=False)
    pass 

def main():
    """
    Main Function for ETL pipeline:
        1. load_data function for data extraction from csv files.
        2. clean data function for data cleaning and formating.
        3. save_data function for data upload into SQLite database.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()