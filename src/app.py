import streamlit as st
import pandas as pd
import joblib
import json
import base64

def set_bg_image(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Use relative path from current working directory (root of repo)
set_bg_image("src/ipl_bg.jpeg")

team_name_map = {
    'Chennai Super Kings': 'Chennai Super Kings',
    'Mumbai Indians': 'Mumbai Indians',
    'Royal Challengers Bangaluru': 'Royal Challengers Bangalore',
    'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Rajasthan Royals': 'Rajasthan Royals',
    'Lucknow Super Giants': 'Lucknow Super Giants',
    'Punjab Kings': 'Kings XI Punjab',
    'Delhi Capitals': 'Delhi Daredevils',
    'Sunrisers Hyderabad': 'Deccan Chargers',
    'Gujarat Titans': 'Gujarat Lions'
}

venue_map = {
    'Chepauk, Chennai': 'MA Chidambaram Stadium, Chepauk',
    'Wankhede, Mumbai': 'Wankhede Stadium',
    'Chinna Swamy, Bangaluru': 'M Chinnaswamy Stadium',
    'Kolkata': 'Eden Gardens',
    'Jaipur': 'Sawai Mansingh Stadium',
    'Arun Jaitley, Delhi': 'Feroz Shah Kotla',
    'Punjab': 'Punjab Cricket Association Stadium',
    'Lucknow': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
    'Ahmedabad': 'Narendra Modi Stadium',
    'Uppal, Hyd': 'Rajiv Gandhi International Stadium, Uppal',
    'Gujarat': 'Sardar Patel Stadium'
}

model_output_to_new_team = {
    'toss_Deccan Chargers': 'Sunrisers Hyderabad',
    'toss_Delhi Daredevils': 'Delhi Capitals',
    'toss_Kings XI Punjab': 'Punjab Kings',
    'toss_Gujarat Lions': 'Gujarat Titans',
    'toss_Royal Challengers Bangalore': 'Royal Challengers Bangaluru',
    'toss_Chennai Super Kings': 'Chennai Super Kings',
    'toss_Mumbai Indians': 'Mumbai Indians',
    'toss_Kolkata Knight Riders': 'Kolkata Knight Riders',
    'toss_Rajasthan Royals': 'Rajasthan Royals',
    'toss_Lucknow Super Giants': 'Lucknow Super Giants',
}

def preprocess_input(user_input, feature_columns):
    df = pd.DataFrame([user_input])
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_encoded

# Fix: paths relative to repo root, where Streamlit runs from
model = joblib.load('models/match_winner_model.pkl')
with open('models/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

st.markdown(
    "<h1 style='color: white;'>🏏 Vamsi's IPL Match Winner Predictor</h1>",
    unsafe_allow_html=True
)

teams = list(team_name_map.keys())
venues = list(venue_map.keys())

team1 = st.selectbox("Select Team 1", options=teams)
team2 = st.selectbox("Select Team 2", options=[team for team in teams if team != team1])
toss_winner = st.selectbox("Select Toss Winner", options=[team1, team2])
toss_decision = st.selectbox("Toss Decision", options=["bat", "field"])
venue = st.selectbox("Venue", options=venues)

if st.button("Predict Winner"):
    user_input = {
        'team1': team_name_map.get(team1, team1),
        'team2': team_name_map.get(team2, team2),
        'toss_winner': team_name_map.get(toss_winner, toss_winner),
        'toss_decision': toss_decision,
        'venue': venue_map.get(venue, venue)
    }

    input_df = preprocess_input(user_input, feature_columns)
    raw_prediction = model.predict(input_df)[0]

    prediction = model_output_to_new_team.get(raw_prediction, raw_prediction)

    st.markdown(
        f"""
        <div style="
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
        ">
            🎯 Predicted Match Winner: <strong>{prediction}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
