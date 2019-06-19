import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# CSWD for metric tracking
import cdsw

import pickle

#Useful Links:

# Timeline (e.g. to estimate age of Jon Snow at end of season 8)
#https://gameofthrones.fandom.com/wiki/Timeline

#comment
#data source
#https://www.kaggle.com/mylesoneill/game-of-thrones#character-predictions.csv

# Derived from
#https://got.show/machine-learning-algorithm-predicts-death-game-of-thrones

# Wiki with character information e.g. correctly spelled names
#https://awoiaf.westeros.org/index.php/Main_Page

# Set columns if character reached a certain age e.g. 
# age 1 => reachedAge0 = 1, reachedAge1 = 1, reachedAge1 = 0 .. 
def encode_age(data_frame, min_age, max_age):
    age_columns = list()
    #N_age = max_age-min_age
    for i in range(min_age,max_age+1):
        age_col = 'reachedAge'+str(i)
        # Set columns to 0 => not alive by default
        data_frame[age_col] = float(0)

        # Set age col to 1 if character passed the age
        data_frame.loc[data_frame['age']>=i, age_col] = 1
        age_columns.append(age_col)
    return data_frame, age_columns


# Since we need the encode age function in our model as well, we differentiate
# between importing and calling this script (main)
def main():
  
  # load prediction data 
  pred_set = pd.read_csv("data/character-predictions_pose.csv", sep=",", header=0)

  ###########################################
  ############ Preprocess Data ##############
  ###########################################
  
  #Filter out NaN for age
  pred_set = pred_set.loc[pred_set['age'].notnull()]

  # add col for whether a character has a title or not
  pred_set.at[pred_set['title'].isnull(), 'has_title'] = 0
  pred_set.at[pred_set['title'].notnull(), 'has_title'] = 1

  # This section creates a binary column for each existing house e.g. Stark and fills them
  houses = pred_set['house'].astype('category')
  houses = pd.get_dummies(houses)

  pred_set = pred_set.join(houses)

  # remove invalid characters with negative age
  pred_set = pred_set.drop(pred_set[pred_set['age']<0].index)

  # checking for negative age
  print(pred_set['age'].min())
  print(pred_set.loc[pred_set['age']<0, 'name'])

  # Create the binary columns for each possible age a character can reach
  max_age = pred_set['age'].max()
  min_age = pred_set['age'].min()
  pred_set, age_columns = encode_age(pred_set, min_age.astype(int), max_age.astype(int))

  # Show rows with characters younger than 25 to validate age encoding
  print(pred_set.loc[pred_set['age']<=25, ['name', 'age'] + age_columns[0:23]])

  # Normalize age column to 0-1 although this column is not needed anymore due to binary columns
  N_age = max_age-min_age
  #pred_set.at[pred_set['age'].notnull(), 'age'] = pred_set.loc[pred_set['age'].notnull(), 'age'].values / N_age
  
  # Popularity index and other columns do not need to be normalized bc they are already between 0-1

  # Show how many columns and rows there are in our preprocessed dataset
  print(pred_set.shape)
  
  # Add number of columns as a metric
  cdsw.track_metric("#columns", pred_set.shape[1])
  
  # Export preprocessed data into pickle file so we can use in model as well
  pred_set.to_pickle("dataset.pkl")
  
  ###########################################
  #### Fit model with preprocessed data #####
  ###########################################
  
  # Pck relevant columns for fitting
  rel_columns = [
    'male',
    'isNoble',
    #'isAliveSpouse',
    #'isMarried',
    #'isAliveHeir',
    #'isAliveMother',
    #'isAliveFather',
    'numDeadRelations',
    'popularity',
    'isPopular',
    'has_title',
    'book1',
    'book2',
    'book3',
    'book4',
    'book5',
    'boolDeadRelations',
    'plod',
    'age'
    ]

  # Add binary columns for houses and ages
  rel_columns += list(houses.columns) + age_columns
  
  # Save relevant columns for model
  pickle.dump(rel_columns, open("columns.pkl", 'wb'), protocol=2)

  # split dataset into training and testing data with scikitlearn function
  trainSet, testSet = train_test_split(pred_set, test_size=0.2, random_state=78)
  
  # Split select input data (X) and output data (Y) for training
  training_setX = trainSet[rel_columns]
  training_labelsY = trainSet['isAlive']

  # Split select input data (X) and output data (Y) for testing
  test_setX = testSet[rel_columns]
  test_labelsY = testSet['isAlive']

  # Convert pandas dataframes into numpy arrays
  trainingXnp = training_setX.to_numpy()
  trainingYnp = training_labelsY.to_numpy()
  testXnp = test_setX.to_numpy()
  testYnp = test_labelsY.to_numpy()

  #Create Neural Network (3 Layer Feedforward Network)
  model = keras.models.Sequential()

  # Add input layer with number of input columns as input and half of that for 
  # input of next layer
  model.add(keras.layers.Dense(int(len(rel_columns)/2), 
                               input_dim=len(rel_columns), 
                               activation='linear')
           )

  # Add one hidden layer and one output with only one node 
  # (represents change of survival)
  model.add(keras.layers.Dense(1, activation='sigmoid'))


  # Compile for a binary classification problem (dies/survives)
  model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train model with training data
  model.fit(trainingXnp, trainingYnp)
  
  # Evaluate model
  eval_out = model.evaluate(testXnp, testYnp)
  cdsw.track_metric("accuracy", float(eval_out[1]))
  print("Accuracy: " + str(eval_out[1]))
  
  # Export model parameters to JSON file
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
      
  # Export fitted weights of model into h5 file
  model.save_weights("model.h5")
  

# If script is not imported run main function
if __name__ == "__main__":
    main()
