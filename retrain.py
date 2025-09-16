import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_excel('Cleaned Data.xlsx')

# Rename columns
df.rename(columns={'I identify as having a mental illness':'MentallyIll'}, inplace=True)
df.rename(columns={'I am unemployed':'Unemployed'}, inplace=True)
df.rename(columns={'Annual income (including any social welfare programs) in USD':'Income'}, inplace=True)

# Replace Yes and No's with 0s and 1s, replace NaNs
df = df.replace(to_replace=['No', 'Yes'], value=[0, 1])
df = df.fillna(0)

# Remove first row of response
df = df.iloc[1:]

# Replace issues with binary representations
df = df.replace(['Lack of concentration','Anxiety', 'Depression',
                   'Obsessive thinking', 'Panic attacks', 'Compulsive behavior',
                   'Mood swings', 'Tiredness'], 1)

# Replace gender -> Male = 1, Female = 0
df = df.replace(to_replace=['Female', 'Male'], value=[0, 1])

# Replace age with median
df = df.replace(to_replace=['> 60', '45-60', '30-44','18-29'], value=[65, 52, 37, 23])

# Replace education with ranking, 0 is lowest educated 7 is highest
df = df.replace(to_replace=['Completed Phd', 'Some Phd', 'Completed Masters','Some\xa0Masters',
                              'Completed Undergraduate', 'Some Undergraduate', 'High School or GED', 'Some highschool']
                              , value=[7, 6, 5, 4, 3, 2, 1, 0])

# Set target and drop from data
target = df['MentallyIll']
df.drop(['MentallyIll'], axis=1, inplace=True)

# Remove last 3 irrelevant columns
df = df.iloc[:, :-3]

X = df

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=0)

forest = RandomForestClassifier(max_depth=10, random_state=0)
forest.fit(X_train, y_train)

print("Model accuracy:", forest.score(X_test, y_test))

# Save model
with open('model.pkl', 'wb') as files:
    pickle.dump(forest, files)

print("Model retrained and saved.")
