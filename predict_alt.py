# This predictor 

import fit

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import pickle

def predict(args):
  
      # Load existing model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    
    # load weights into new model
    model.load_weights("model.h5")
    
    # Load the preprocessed dataset
    pred_set = pd.read_pickle("./dataset.pkl")
    
    
    # Change the Name here to perform queries for other characters e.g. "Jon Snow"
    # You can choose characters (with right spelling) from this wiki: 

    # Load relevant columns from pickle file (created by fit.py)
    rel_columns = pickle.load(open('columns.pkl', 'rb'))
    


    
    # Run as model
    print("Run as model")
      
    # Get character name from input args
    character_name = str(args.get('name'))
      
    # create subset dataframe for character
    character = pred_set[pred_set['name'] == character_name]
      
    character['age'] = float(args.get('age'))
      
    character['popularity'] = float(args.get('popularity'))
      
    character['numDeadRelations'] = int(args.get('numDeadRelations'))
      
    # encode age of character
    character, age_columns = fit.encode_age(character, 0, 100)
      
    # Output a few columns of the selected character for validation
    print(character.loc[pred_set['name'] == character_name, 
                         ['name', 'age','popularity', 'isAlive', 'numDeadRelations']])

      
    # select relevant columns of character dataset
    character = character[rel_columns]
    
    # Turn character dataset into input vector
    predX = character.to_numpy()
      
    return float(model.predict(predX)[0])