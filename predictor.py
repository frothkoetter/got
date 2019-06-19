import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=4)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import pdb

#Useful Links:

# Timeline (e.g. to estimate age of Jon Snow at end of season 8)
#https://gameofthrones.fandom.com/wiki/Timeline

#data source
#https://www.kaggle.com/mylesoneill/game-of-thrones#character-predictions.csv

# Derived from
#https://got.show/machine-learning-algorithm-predicts-death-game-of-thrones

# Wiki with character information e.g. correctly spelled names
#https://awoiaf.westeros.org/index.php/Main_Page

def encode_age(data_frame, min_age, max_age):
    for i in range(min_age,max_age+1):
        age_col = 'reachedAge'+str(i)
        # Set columns to 0 => not alive by default
        data_frame[age_col] = float(0)

        # Set age col to 1 if character passed the age
        data_frame.loc[data_frame['age']>=i, age_col] = 1
        age_columns.append(age_col)
    return data_frame



pred_set = pd.read_csv("data/character-predictions_pose.csv", sep=",", header=0)


#pred_set.columns['has_title'] = 0

#pred_set['has_title'] = 

#print(pred_set)

#Filter out NaN for age
#pp.pprint(list(pred_set.loc[pred_set['age'].isnull(), 'name']))
pred_set = pred_set.loc[pred_set['age'].notnull()]

# add col for wether a character has a title or not
pred_set.at[pred_set['title'].isnull(), 'has_title'] = 0
pred_set.at[pred_set['title'].notnull(), 'has_title'] = 1

print(list(pred_set.columns))

houses = pred_set['house'].astype('category')
houses = pd.get_dummies(houses)

pred_set = pred_set.join(houses)

print(pred_set['age'].min())
print(pred_set.loc[pred_set['age']<0, 'name'])

# remove invalid characters with negative age
pred_set = pred_set.drop(pred_set[pred_set['age']<0].index)

print(pred_set['age'].min())
print(pred_set.loc[pred_set['age']<0, 'name'])

age_columns = list()

print(pred_set.shape)

max_age = pred_set['age'].max().astype(int)
min_age = pred_set['age'].min().astype(int)

# create a column for each characters age by which she was still alive => encoding of age
pred_set = encode_age(pred_set, min_age,max_age)



print(pred_set.loc[pred_set['age']<=25, ['name', 'age'] + age_columns[0:23]])

# Normalize Age to 0-1

# max_age = pred_set['age'].max()
# min_age = pred_set['age'].min()
# N_age = max_age-min_age

# pred_set.at[pred_set['age'].notnull(), 'age'] = pred_set.loc[pred_set['age'].notnull(), 'age'].values / N_age

# Normalize Popularity
max_val = pred_set['popularity'].max()
min_val = pred_set['popularity'].min()
N = max_val-min_val

#pred_set.at[pred_set['popularity'].notnull(), 'popularity'] = pred_set.loc[pred_set['popularity'].notnull(), 'popularity'].values / N

#pred_set['popularity'] = pred_set['popularity'].astype(int)

#print(pred_set['age'])

print(pred_set.shape)


dead_characters = pred_set.loc[pred_set['isAlive']==0]

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

rel_columns += houses.columns.to_list() + age_columns

trainSet, testSet = train_test_split(pred_set, test_size=0.2, random_state=78)

training_setX = trainSet[rel_columns]
training_labelsY = trainSet['isAlive']

test_setX = testSet[rel_columns]
test_labelsY = testSet['isAlive']

print(training_setX.shape)
print(len(rel_columns))

#exit()

trainingXnp = training_setX.to_numpy()

trainingYnp = training_labelsY.to_numpy()

testXnp = test_setX.to_numpy()
testYnp = test_labelsY.to_numpy()

print(testYnp)

use_existing_nn = True

if (use_existing_nn):
    # Load existing model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
else:
    #Create Neural Network
    model = keras.models.Sequential()

    # Add input layer
    model.add(keras.layers.Dense(int(len(rel_columns)/2), input_dim=len(rel_columns), activation='linear'))

    # Add one hidden layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))


    # Compile for a binary classification problem (dies/survives)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(trainingXnp, trainingYnp)
    print(model.evaluate(testXnp, testYnp))

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")


# From here I performed an example of the usage of the predictor
# It predicts the probability of survival for the current age until 100 and plots it

# Change the Name here to perform queries for other characters e.g. "Jon Snow"
# You can choose characters (with right spelling) from this wiki: 

#print(pred_set.loc[pred_set['isAlive']==0,['name','popularity']].to_string())

character_name = "Rhaegar Targaryen"

print(pred_set.loc[pred_set['name'] == character_name, ['name', 'age','popularity', 'isAlive']])

character = pred_set[pred_set['name'] == character_name]

#character['male'] = 1

character = character[rel_columns]

predX = character.to_numpy()

print("Chance of Survival: ", model.predict(predX)[0])

#print(character['male'])

x_ages = list()
y_prob = list()

for i in range(int(character['age'].values[0]), 100):

    character['age'] = float(i)

    character = encode_age(character, min_age, max_age)

    character = character[rel_columns]

    predX = character.to_numpy()

    x_ages.append(i)
    y_prob.append(float(model.predict(predX)[0]))

model = None
character = None
pred_set = None

plt.plot(x_ages, y_prob)
plt.show()


# Observations: 
# Just increasing the age usually keeps the probability on approx. the same level until they get
# older (~70) and the chance of deatch increases rapidly

# The number of dead relations has a pretty high influence on the survival chance => If many people
# close to you die then its more likely that you will die too.










