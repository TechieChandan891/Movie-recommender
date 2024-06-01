import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
world_cup = pd.read_csv('World_cup_2023.csv')
results = pd.read_csv('results.csv')
ranking = pd.read_csv('Icc_ranking.csv')
fixtures = pd.read_csv('Fixtures.csv')

# Your existing data processing and model training code here...

# Function to get predictions
def get_predictions():
    pred_set = []
    for index, row in fixtures.iterrows():
        if row['first_position'] < row['second_position']:
            pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'winning_team': None})
        else:
            pred_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'winning_team': None})
    
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set
    pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

    missing_cols = set(final.columns) - set(pred_set.columns)
    for c in missing_cols:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    pred_set = pred_set.drop(['Winner'], axis=1)
    predictions = rf.predict(pred_set)

    result_list = []
    for i in range(fixtures.shape[0]):
        result_dict = {
            'Team_1': backup_pred_set.iloc[i, 1],
            'Team_2': backup_pred_set.iloc[i, 0],
            'Winner': backup_pred_set.iloc[i, 1] if predictions[i] == 1 else backup_pred_set.iloc[i, 0]
        }
        result_list.append(result_dict)

    return result_list

# Streamlit app
st.title("Cricket World Cup 2023 Predictions")

# Display predictions
predictions = get_predictions()
for prediction in predictions:
    st.write(f"{prediction['Team_1']} vs {prediction['Team_2']} - Winner: {prediction['Winner']}")
