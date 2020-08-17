# Importing necessary libraries
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle

# Reading the data
df = pd.read_csv("absenteeism.csv")
#print(absenteeism.head())

df.drop("ID", axis=1, inplace = True)
#df.columns = df.columns.str.replace('/', 'per').str.strip()

df_features = df[['Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', #'Transportation expense',
        #'Distance from Residence to Work', 'Service time', 'Age', #'Work load Average/day',
                  # 'Hit target',
        'Disciplinary failure', 'Education', 'Son', 'Social drinker', 'Social smoker', 'Pet', 'Weight',
        'Height', 'Body mass index' ]]
X = df_features
y = df['Absenteeism time in hours']

# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = RandomForestRegressor(n_estimators=1000, random_state=0, oob_score=True, n_jobs=-1)
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb'))