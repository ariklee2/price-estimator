import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

#Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from google.colab import drive


drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/Kaggle/AB_NYC_2019.csv')
print("Shape:", df.shape)
display(df.head())

"""Understanding the Data"""

print("\nINFO ")
df.info()

print("\nUNIQUE COUNTS PER COLUMN")
display(df.nunique())

print("\nSAMPLE OF KEY COLUMNS")
cols = ['name','host_name','neighbourhood_group','neighbourhood',]
existing = [c for c in cols if c in df.columns]
display(df[existing].head(10))

print("\nBorough counts:")
print(df['neighbourhood_group'].value_counts(dropna=False))

print("\nRoom type counts:")
print(df['room_type'].value_counts(dropna=False))

"""EDA before cleaning"""

print("\nMISSING VALUES (count)")
display(df.isnull().sum())

#Price distribution
sns.histplot(df['price'], bins=50)
plt.title("Price Distribution (raw)")
plt.show()

#Room type counts
sns.countplot(x='room_type', data=df)
plt.title("Room Types (raw)")
plt.show()

#Price by borough
sns.boxplot(x='neighbourhood_group', y='price', data=df)
plt.title("Price by Borough (raw)")
plt.show()

"""Cleaning"""

#Drop exact duplicates
df.drop_duplicates(inplace=True)

#Convert dates safely (errors -> NaT)
if 'last_review' in df.columns:
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

#Fill simple missing values
#If no reviews, set reviews_per_month to 0 (common beginner choice)
if 'reviews_per_month' in df.columns:
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

#Remove obviously bad prices (<= 0)
if 'price' in df.columns:
    df = df[df['price'] > 0]

#Gentle outlier trim on price (cut above 99th percentile)
if 'price' in df.columns:
    p99 = df['price'].quantile(0.99)
    df = df[df['price'] <= p99]

print("Post-clean shape:", df.shape)
display(df.isnull().sum())

"""Visuals after cleaning"""

sns.histplot(df['price'], bins=50)
plt.title("Price Distribution (cleaned)")
plt.show()

sns.countplot(x='room_type', data=df)
plt.title("Room Types (cleaned)")
plt.show()

sns.boxplot(x='neighbourhood_group', y='price', data=df)
plt.title("Price by Borough (cleaned)")
plt.show()

print("\nAverage price by borough (cleaned):")
display(df.groupby('neighbourhood_group')['price'].mean().round(2).sort_values(ascending=False))

print("\nAverage price by room_type (cleaned):")
display(df.groupby('room_type')['price'].mean().round(2).sort_values(ascending=False))

"""#Cleaning the Data - Part 2"""

#Isolate the data we actually want to use
cols = [
    'price',
    'neighbourhood_group',
    'room_type',
    'latitude',
    'longitude',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'
]

#Keep only those columns that exist in our dataframe
existing = [c for c in cols if c in df.columns]
df_model = df[existing].dropna()
print("\nColumns kept for modeling:")
print(existing)
print("Shape after selecting columns:", df_model.shape)
display(df_model.head())

#Check the data types and missing values again
print("\nData Types")
print(df_model.dtypes)

print("\nMissing Values")
print(df_model.isnull().sum())

"""Define x and y"""

#Make sure df_model exists
try:
    print("df_model found. Shape:", df_model.shape)
except NameError:
    raise RuntimeError("df_model not found. Run your Week 3 code first to create df_model.")

#Define Features (x) and Target (y)
target = 'price'

#Columns that will be used as features
features = [
    'neighbourhood_group', 'room_type', 'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365'
]

#Create x (features) and y (target)
X = df_model[features]
y = df_model[target]

print("\nFeatures and target defined.")
print("X shape:", X.shape)
print("y shape:", y.shape)

"""Train/Test Split"""

from sklearn.model_selection import train_test_split

#Split 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData split complete:")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

#Preview to confirm split worked
print("\nTraining set preview:")
display(X_train.head())

print("\nTesting set preview:")
display(X_test.head())

"""First Model: Linear Regression"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

#Identify which columns are categorical (the ones with strings)
cat_cols = ['neighbourhood_group', 'room_type']

#Preprocessing: One-hot encode only the categorical columns
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'   #keep numeric columns as they are
)

#Build pipeline: preprocess → linear regression model
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('linreg', LinearRegression())
])

#Train the model
model.fit(X_train, y_train)
print("\nModel training complete.")

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance (Linear Regression):")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

#Show sample predictions
results_preview = pd.DataFrame({
    'Actual Price': y_test[:10].values,
    'Predicted Price': y_pred[:10]
})

print("\nSample Predictions (first 10):")
display(results_preview)

sample = X_test.head(10).copy()
sample['Actual Price'] = y_test[:10].values
sample['Predicted Price'] = y_pred[:10]

display(sample)

"""KNN Regression"""

#Second Model: K-Nearest Neighbors (KNN) Regression
#Uses the same X_train, X_test, y_train, y_test, and `preprocess` as before.

from sklearn.neighbors import KNeighborsRegressor

#Build pipeline: preprocess → KNN model
knn_model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('knn', KNeighborsRegressor(n_neighbors=5))
])

#Train the model
knn_model.fit(X_train, y_train)
print("\nKNN model training complete.")

#Make predictions
y_pred_knn = knn_model.predict(X_test)

#Evaluate the model
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("\nModel Performance (KNN Regression):")
print("MAE:", round(mae_knn, 2))
print("RMSE:", round(rmse_knn, 2))
print("R² Score:", round(r2_knn, 4))

#Show sample predictions
results_preview_knn = pd.DataFrame({
    'Actual Price': y_test[:10].values,
    'Predicted Price (KNN)': y_pred_knn[:10]
})

print("\nSample Predictions (first 10) – KNN:")
display(results_preview_knn)

sample_knn = X_test.head(10).copy()
sample_knn['Actual Price'] = y_test[:10].values
sample_knn['Predicted Price (KNN)'] = y_pred_knn[:10]

display(sample_knn)

"""Decision Tree Regression"""

#Third Model: Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

#Build pipeline: preprocess → Decision Tree model
tree_model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('tree', DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ))
])

#Train the model
tree_model.fit(X_train, y_train)
print("\nDecision Tree model training complete.")

#Make predictions
y_pred_tree = tree_model.predict(X_test)

# Evaluate the model
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("\nModel Performance (Decision Tree Regression):")
print("MAE:", round(mae_tree, 2))
print("RMSE:", round(rmse_tree, 2))
print("R² Score:", round(r2_tree, 4))

#Show sample predictions
results_preview_tree = pd.DataFrame({
    'Actual Price': y_test[:10].values,
    'Predicted Price (Tree)': y_pred_tree[:10]
})

print("\nSample Predictions (first 10) – Decision Tree:")
display(results_preview_tree)

sample_tree = X_test.head(10).copy()
sample_tree['Actual Price'] = y_test[:10].values
sample_tree['Predicted Price (Tree)'] = y_pred_tree[:10]

display(sample_tree)

"""Price Predictor"""

#Make sure needed objects exist
try:
    df_model, features, model, knn_model, tree_model
except NameError as e:
    raise RuntimeError("Make sure you have run all the cells above (df_model, features, model, knn_model, tree_model).") from e

#Get valid options for the categorical columns
valid_neighbourhoods = sorted(df_model['neighbourhood_group'].dropna().unique())
valid_room_types = sorted(df_model['room_type'].dropna().unique())

#Pre-compute average latitude/longitude per neighbourhood_group
avg_coords = df_model.groupby('neighbourhood_group')[['latitude', 'longitude']].mean()

print("Welcome to the Airbnb Price Predictor")
print("\nValid neighbourhood_group options:")
for n in valid_neighbourhoods:
    print(" -", n)

print("\nValid room_type options:")
for r in valid_room_types:
    print(" -", r)

print("\nType 'quit' at any time for neighbourhood_group to exit.\n")

#Small helper to safely read numeric input
def get_number(prompt, min_value=None, max_value=None):
    while True:
        text = input(prompt).strip()
        try:
            value = float(text)
            if min_value is not None and value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Please enter a value <= {max_value}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number (e.g. 3 or 3.5).")

#Main loop
while True:
    # --- Categorical inputs ---
    neigh = input("\nEnter neighbourhood_group (or 'quit' to stop): ").strip()
    if neigh.lower() == "quit":
        print("Exiting price predictor. Goodbye!")
        break

    if neigh not in valid_neighbourhoods:
        print("Invalid neighbourhood_group. Please choose from the list shown above.")
        continue

    room = input("Enter room_type: ").strip()
    if room not in valid_room_types:
        print("Invalid room_type. Please choose from the list shown above.")
        continue

    #Use the average latitude/longitude for that neighbourhood
    lat = avg_coords.loc[neigh, 'latitude']
    lon = avg_coords.loc[neigh, 'longitude']

    # --- Numeric inputs ---
    min_nights = get_number("Enter minimum_nights (>= 1): ", min_value=1)
    num_reviews = get_number("Enter number_of_reviews (>= 0): ", min_value=0)
    reviews_pm = get_number("Enter reviews_per_month (>= 0): ", min_value=0)
    host_count = get_number("Enter calculated_host_listings_count (>= 0): ", min_value=0)
    avail_365 = get_number("Enter availability_365 (0–365): ", min_value=0, max_value=365)

    #Build a single-row DataFrame for prediction
    input_data = pd.DataFrame([{
        'neighbourhood_group': neigh,
        'room_type': room,
        'latitude': lat,
        'longitude': lon,
        'minimum_nights': min_nights,
        'number_of_reviews': num_reviews,
        'reviews_per_month': reviews_pm,
        'calculated_host_listings_count': host_count,
        'availability_365': avail_365
    }])

    #Make sure the columns are in the same order as the features we trained on
    input_data = input_data[features]

    #Get predictions from all three models
    pred_lin = model.predict(input_data)[0]
    pred_knn = knn_model.predict(input_data)[0]
    pred_tree = tree_model.predict(input_data)[0]

    #Show results (Decision Tree is the main one)
    print("\n--- Predicted Nightly Price (USD) ---")
    print(f"Decision Tree (best):      ${pred_tree:,.2f}")
    print(f"Linear Regression (middle): ${pred_lin:,.2f}")
    print(f"KNN (worst):                ${pred_knn:,.2f}")