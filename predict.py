# This predictor 

import fit

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

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
    


    # If run in sessing then args is None
    if (args == None):
      
      print("Run in Session")
      
      # Change the Name here to perform queries for other characters 
      # e.g. "Jon Snow". You can choose characters (with right spelling) from 
      # the wiki of ice and fire:
      character_name = "Rhaegar Targaryen"

      # create subset dataframe for character
      character = pred_set[pred_set['name'] == character_name]
      
      character = character[rel_columns]
      
      # Output a few columns of the selected character
      print(pred_set.loc[pred_set['name'] == character_name, ['name', 'age','popularity', 'isAlive']])          

      # Output the current chance of survival (based on training dataset)
      predX = character.to_numpy()
      print("Chance of Survival: ", model.predict(predX)[0])

      x_ages = list()
      y_prob = list()
        
      # Predict change of survival for each age from the current age to 100
      for i in range(int(character['age'].values[0]), 100):

          character['age'] = float(i)

          # We hardcode the age here since we know it, usually it would be better to
          # use the dataset
          character, age_columns = fit.encode_age(character, 0, 100)

          character = character[rel_columns]

          predX = character.to_numpy()

          x_ages.append(i)
          y_prob.append(float(model.predict(predX)[0]))

      # delete keras model and pandas data
      model = None
      character = None
      pred_set = None

      # plot results of age prediction
      plt.plot(x_ages, y_prob)
      plt.show()
      
      return 0
    
    else:
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