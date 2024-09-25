#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


@st.cache_resource
def load_data():
    pickle_file_path = r'C:\Users\spenc\VictorVis\CleanData\solo_players_df.pkl'
    with open(pickle_file_path, 'rb') as file:
        return pickle.load(file)

solo_players_df = load_data()


# In[3]:


print(solo_players_df.columns)


# In[13]:


solo_players = solo_players_df[['team', 'real_name', 'kd_ratio', 'kpr', 'kills_per_round',
                                  'firepower_score', 'firepower_rounds_with_kill',
                                  'impact', 'firepower_kills_per_round', 'firepower_rounds_with_multi_kill',
                                  'opening_success', 'kast']]

solo_players.set_index('team', inplace=True)
solo_players


# In[15]:


X = solo_players.drop(['real_name', 'kast'], axis=1)  # Assuming 'team_name' is in your dataset
y = solo_players['kast']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

if y.dtype == 'object' or y.nunique() < 10:  # Assuming it's categorical if less than 10 unique values
    print("\n'real_name' appears to be categorical. Using RandomForestClassifier.")
    # Encode categorical target
    le = LabelEncoder()
    y = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                   min_samples_split=2, min_samples_leaf=1, 
                                   random_state=42)
else:
    print("\n'real_name' appears to be continuous. Using RandomForestRegressor.")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                  min_samples_split=2, min_samples_leaf=1, 
                                  random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Streamlit app
st.title("Team Performance Predictor")

# Create two columns for team inputs
col1, col2 = st.columns(2)

# Function to create input fields for a team
def team_input(column, team_name):
    with column:
        st.subheader(f"{team_name} Stats")
        team_data = {}
        for feature in X.columns:
            team_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", 
                                                 min_value=0.0, 
                                                 max_value=1.0, 
                                                 value=0.5, 
                                                 key=f"{team_name}_{feature}")
        return pd.DataFrame([team_data])

# Get input for both teams
team1_data = team_input(col1, "Team 1")
team2_data = team_input(col2, "Team 2")

if st.button("Predict Winner"):
    # Make predictions
    team1_pred = model.predict(team1_data)[0]
    team2_pred = model.predict(team2_data)[0]

    # Display results
    st.subheader("Prediction Results")
    st.write(f"Team 1 predicted KAST: {team1_pred:.2f}")
    st.write(f"Team 2 predicted KAST: {team2_pred:.2f}")

    if team1_pred > team2_pred:
        st.success("Team 1 is predicted to win!")
    elif team2_pred > team1_pred:
        st.success("Team 2 is predicted to win!")
    else:
        st.info("It's a tie based on the predictions!")

    # Display feature importances
    st.subheader("Feature Importances")
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    st.bar_chart(importances.set_index('feature'))

st.sidebar.info("This app predicts the winner between two teams based on their performance metrics. Enter the stats for each team and click 'Predict Winner' to see the results.")


# ##RandomForest Regressor 
# # Create a RandomForestRegressor model
# model = RandomForestRegressor(n_estimators=128, random_state=1)
# 
# # Fit the model
# model.fit(X_train, y_train)
# 
# # Make predictions
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# 
# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")

# In[9]:


# Create a RandomForestRegressor model
#model = RandomForestRegressor(n_estimators=128, random_state=1)

# Fit the model
#model.fit(X_train, y_train)

# Make predictions
#y_pred = model.predict(X_test)
#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#print(f"Mean Squared Error: {mse}")
#print(f"R^2 Score: {r2}")

