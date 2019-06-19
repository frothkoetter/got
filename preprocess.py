import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Useful Links:

# Timeline (e.g. to estimate age of Jon Snow at end of season 8)
#https://gameofthrones.fandom.com/wiki/Timeline

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
    for i in range(min_age,max_age+1):
        age_col = 'reachedAge'+str(i)
        # Set columns to 0 => not alive by default
        data_frame[age_col] = float(0)

        # Set age col to 1 if character passed the age
        data_frame.loc[data_frame['age']>=i, age_col] = 1
        age_columns.append(age_col)
    return data_frame, age_columns

  
def main():
  
  # load prediction data 
  pred_set = pd.read_csv("data/character-predictions_pose.csv", sep=",", header=0)

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
  pred_set.at[pred_set['age'].notnull(), 'age'] = pred_set.loc[pred_set['age'].notnull(), 'age'].values / N_age
  
  # Popularity index and other columns do not need to be normalized bc they are already between 0-1

  # Show how many columns and rows there are in our preprocessed dataset
  print(pred_set.shape)
  
  # Export data into pickle file
  pred_set.to_pickle("dataset.pkl", protocol=3)
  


# If script is not imported run preprocess routine
if __name__ == "__main__":
    main()



