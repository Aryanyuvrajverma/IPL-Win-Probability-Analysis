import pandas as pd
import numpy as np


# 1. LOAD DATA

df = pd.read_excel('D:/project/ipl.xlsx')


# 2. DATA CLEANING

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['date'] = pd.to_datetime(df['date'])

df['total_runs'] = df['runs_scored'] + df['extras']
df['is_wicket'] = df['player_out'].notnull().astype(int)

df['player_out'] = df['player_out'].fillna('Not Out')
df['wicket_kind'] = df['wicket_kind'].fillna('No Wicket')
df['fielder'] = df['fielder'].fillna('No Fielder')

df.drop_duplicates(inplace=True)


# 3. FILTER 2ND INNINGS

df_2nd = df[df['inning'] == 2].copy()

# Target variable
df_2nd['win'] = df_2nd.apply(
    lambda x: 1 if x['batting_team'] == x['winner'] else 0,
    axis=1
)

# 4. FEATURE ENGINEERING

df_2nd['pressure_index'] = df_2nd.apply(
    lambda x: x['required_run_rate'] / x['current_run_rate'] if x['current_run_rate'] != 0 else 0,
    axis=1
)

df_2nd['run_rate_diff'] = df_2nd['current_run_rate'] - df_2nd['required_run_rate']
df_2nd['balls_used'] = 120 - df_2nd['balls_remaining']
df_2nd['wickets_lost'] = 10 - df_2nd['wickets_remaining']
df_2nd['overs_completed'] = df_2nd['balls_used'] / 6

# Handle inf and NaN
df_2nd.replace([np.inf, -np.inf], np.nan, inplace=True)
df_2nd.dropna(inplace=True)


# 5. FEATURE SELECTION

features = [
    'batting_team',
    'bowling_team',
    'venue',
    'balls_remaining',
    'wickets_remaining',
    'balls_used',
    'wickets_lost',
    'overs_completed'
]


# 6. TRAIN-TEST SPLIT (MATCH LEVEL)

from sklearn.model_selection import train_test_split

match_ids = df_2nd['match_id'].unique()

train_matches, test_matches = train_test_split(
    match_ids, test_size=0.2, random_state=42
)

train_df = df_2nd[df_2nd['match_id'].isin(train_matches)]
test_df = df_2nd[df_2nd['match_id'].isin(test_matches)]

X_train = train_df[features]
y_train = train_df['win']

X_test = test_df[features]
y_test = test_df['win']

# Convert categorical → numerical
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 7. RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

# Feature Importance
importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)
print("\nTop Features:\n", importance.sort_values(ascending=False).head(10))

# 8. LOGISTIC REGRESSION (WITH SCALING)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, lr_pred))

# 9. PROBABILITY OUTPUT (OPTIONAL)

probs = lr_model.predict_proba(X_test_scaled)
print("\nSample Probabilities:\n", probs[:5])

# 10. FINAL DATA CHECK

print("\nNull Values Check:\n", df_2nd.isnull().sum())


import matplotlib.pyplot as plt

# 1. BAR CHART (Top Teams)

plt.figure(figsize=(8,5))
df['winner'].value_counts().head(10).plot(kind='bar')
plt.title("Top Teams by Wins")
plt.xlabel("Teams")
plt.ylabel("Wins")
plt.xticks(rotation=45)
plt.show()


# 2. PIE CHART (Toss Decision)

plt.figure(figsize=(6,6))
df['toss_decision'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Toss Decision Distribution")
plt.ylabel("")
plt.show()


# 3. LINE CHART (Win vs Wickets Remaining)

plt.figure(figsize=(8,5))
df_2nd.groupby('wickets_remaining')['win'].mean().plot()
plt.title("Win Probability vs Wickets Remaining")
plt.xlabel("Wickets Remaining")
plt.ylabel("Win Probability")
plt.show()


# 4. HISTOGRAM (Runs Distribution)

plt.figure(figsize=(8,5))
df['total_runs'].plot(kind='hist', bins=30)
plt.title("Distribution of Runs per Ball")
plt.xlabel("Runs")
plt.show()


# 5. SCATTER PLOT (Balls vs Wickets)

plt.figure(figsize=(8,5))
plt.scatter(df_2nd['balls_remaining'], df_2nd['wickets_remaining'], alpha=0.3)
plt.title("Balls Remaining vs Wickets Remaining")
plt.xlabel("Balls Remaining")
plt.ylabel("Wickets Remaining")
plt.show()


# 6. BOX PLOT (Runs by Phase)

df['phase'] = df['over'].apply(lambda x: 'Powerplay' if x <= 6 else ('Middle' if x <= 15 else 'Death'))

plt.figure(figsize=(8,5))
df.boxplot(column='total_runs', by='phase')
plt.title("Runs Distribution by Match Phase")
plt.suptitle("")
plt.xlabel("Match Phase")
plt.ylabel("Runs")
plt.show()


# 7. BAR CHART (Top Batsmen)

plt.figure(figsize=(8,5))
df.groupby('batter')['runs_scored'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top Batsmen")
plt.xticks(rotation=45)
plt.show()


# 8. BAR CHART (Top Bowlers)

plt.figure(figsize=(8,5))
df.groupby('bowler')['is_wicket'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top Bowlers")
plt.xticks(rotation=45)
plt.show()

df.to_csv('D:/project/final_ipl_data.csv', index=False)
