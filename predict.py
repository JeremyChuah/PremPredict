import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df_train = pd.read_csv("train.csv", index_col = 0)
df_test = pd.read_csv("test.csv", index_col = 0)

#create predictors for the model
df_train["Date"] = pd.to_datetime(df_train["Date"])
df_test["Date"] = pd.to_datetime(df_test["Date"])

df_train["venue_code"] = df_train["Venue"].astype("category").cat.codes
df_test["venue_code"] = df_test["Venue"].astype("category").cat.codes

df_train["opp_code"] = df_train["Opponent"].astype("category").cat.codes
df_test["opp_code"] = df_test["Opponent"].astype("category").cat.codes

df_train["hour"] = df_train["Time"].str.replace(":.+", "", regex=True).astype("int")
df_test["hour"] = df_test["Time"].str.replace(":.+", "", regex=True).astype("int")

df_train["day_code"] = df_train["Date"].dt.dayofweek
df_test["day_code"] = df_test["Date"].dt.dayofweek

#rolling predictors from last seasons to current season
df_train['win'] = df_train['Result'].apply(lambda x: 1 if x == 'W' else 0)
df_train['rolling_win_rate'] = df_train.groupby('Team')['win'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
team_rolling_win_rate = df_train.groupby('Team')['rolling_win_rate'].last().reset_index()
team_rolling_win_rate.rename(columns={'rolling_win_rate': 'rolling_win_rate_last'}, inplace=True)
df_test = df_test.merge(team_rolling_win_rate, on='Team', how='left')
df_test['rolling_win_rate'] = df_test['rolling_win_rate_last']
df_test.drop(columns=['rolling_win_rate_last'], inplace=True)
df_test['rolling_win_rate'].fillna(0, inplace=True)

df_train['goals_scored'] = df_train['GF'].astype(int)
df_train['rolling_avg_goals'] = df_train.groupby('Team')['goals_scored'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
team_rolling_avg_goals = df_train.groupby('Team')['rolling_avg_goals'].last().reset_index()
team_rolling_avg_goals.rename(columns={'rolling_avg_goals': 'last_rolling_avg_goals'}, inplace=True)
df_test = df_test.merge(team_rolling_avg_goals, on='Team', how='left', suffixes=('', '_last'))
df_test['rolling_avg_goals'] = df_test['last_rolling_avg_goals']
df_test.drop(columns=['last_rolling_avg_goals'], inplace=True)
df_test.loc[:, 'rolling_avg_goals'] = df_test['rolling_avg_goals'].fillna(0)

df_train['goals_against'] = df_train['GA'].astype(int)
df_train['rolling_avg_ga'] = df_train.groupby('Team')['goals_against'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
team_rolling_avg_ga = df_train.groupby('Team')['rolling_avg_ga'].last().reset_index()
team_rolling_avg_ga.rename(columns={'rolling_avg_ga': 'last_rolling_avg_ga'}, inplace=True)
df_test = df_test.merge(team_rolling_avg_ga, on='Team', how='left', suffixes=('', '_last'))
df_test['rolling_avg_ga'] = df_test['last_rolling_avg_ga']
df_test.drop(columns=['last_rolling_avg_ga'], inplace=True)
df_test.loc[:, 'rolling_avg_ga'] = df_test['rolling_avg_ga'].fillna(0)

predictors = ["venue_code", "opp_code", "hour", "day_code", "rolling_win_rate", 'rolling_avg_goals', 'rolling_avg_ga']
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)
result_mapping = {'W': 2, 'D': 1, 'L': 0}
df_train["target"] = df_train["Result"].map(result_mapping)

#Train the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train = df_train[predictors]
y_train = df_train["target"]
X_test = df_test[predictors]
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_split, y_train_split)

y_val_pred = rf_model.predict(X_val_split)

print("Validation Accuracy:", accuracy_score(y_val_split, y_val_pred))

#predict and export the results
y_test_pred = rf_model.predict(X_test)
df_test['predicted_result'] = y_test_pred
result_mapping_reverse = {2: 'W', 1: 'D', 0: 'L'}
df_test['predicted_result'] = df_test['predicted_result'].map(result_mapping_reverse)
columns_to_save = ['Venue', 'Opponent', 'Team', 'Time', 'predicted_result']
df_test.to_csv('predicted_results.csv', columns=columns_to_save, index=False)