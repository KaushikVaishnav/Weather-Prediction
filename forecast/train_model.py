import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # preferred for scikit-learn models
import os

# Load data
csv_path = 'C:\\Users\\OM\\OneDrive\\Desktop\\MLPROJECT\\weatherproject\\weather.csv'
df = pd.read_csv(csv_path).dropna().drop_duplicates()

# Encode categorical variables
le_wind = LabelEncoder()
df['WindGustDir'] = le_wind.fit_transform(df['WindGustDir'])

le_rain = LabelEncoder()
df['RainTomorrow'] = le_rain.fit_transform(df['RainTomorrow'])

# Features and target
X = df[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']]
y = df['RainTomorrow']

# Train/test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and label encoders
joblib.dump(model, 'rain_model.joblib')
joblib.dump(le_wind, 'le_wind.joblib')
joblib.dump(le_rain, 'le_rain.joblib')

print("Training complete and model saved.")
