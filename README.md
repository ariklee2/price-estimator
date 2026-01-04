# Airbnb Price Estimator (NYC 2019)

## Project Overview
This project builds a **machine learning price estimator** for Airbnb listings in New York City using the **AB_NYC_2019** dataset from Kaggle.  
The goal is to predict a reasonable **nightly price** based on listing characteristics such as location, room type, availability, and reviews.

Three regression models are trained and compared:
- **Linear Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Regression** (best-performing model)

An interactive **command-line price predictor** allows users to input listing details and receive price predictions from all three models.

**Note: All of this was coded in Google Colab, I provived both the python and colab file.

---

## Dataset
- **Source:** Kaggle – *New York City Airbnb Open Data (2019)*
- **File:** `AB_NYC_2019.csv`
- **Target Variable:** `price`

### Features Used
- `neighbourhood_group`
- `room_type`
- `latitude`
- `longitude`
- `minimum_nights`
- `number_of_reviews`
- `reviews_per_month`
- `calculated_host_listings_count`
- `availability_365`

---

## Data Cleaning & Preprocessing
The following steps were applied:
- Removed duplicate rows
- Converted `last_review` to datetime format
- Filled missing `reviews_per_month` values with `0`
- Removed listings with price ≤ 0
- Trimmed extreme outliers above the 99th percentile
- One-hot encoded categorical variables
- Left numeric variables unscaled for tree-based modeling

---

## Exploratory Data Analysis (EDA)
EDA was performed to understand pricing patterns and data quality, including:
- Price distributions (before and after cleaning)
- Price comparison by borough
- Price comparison by room type
- Room type and borough frequency counts

These insights helped guide feature selection and outlier handling.

---

## Models Trained
| Model | Description |
|------|------------|
| Linear Regression | Baseline regression model |
| KNN Regression | Distance-based model |
| Decision Tree Regression | Captures non-linear relationships (best results) |

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Results
The **Decision Tree Regressor** achieved the best performance overall, with:
- Lower MAE and RMSE
- Higher R² score than Linear Regression and KNN

Because of this, the Decision Tree model is used as the **primary predictor** in the final application.

---

## Interactive Price Predictor
The final section of the notebook includes an interactive **command-line tool** that:
1. Prompts the user for listing details
2. Automatically assigns latitude and longitude using borough averages
3. Predicts a nightly price using all three models
4. Displays predictions for easy comparison

---

## How to Run
1. Open the notebook in **Google Colab**
2. Upload `AB_NYC_2019.csv` to your Google Drive
3. Update the file path if necessary
4. Run all cells from top to bottom
5. Use the interactive price predictor at the end of the notebook

---

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Google Colab Drive API

---

## Notes
- This project is for **educational purposes**
- Predicted prices are estimates, not market recommendations
- Dataset reflects NYC Airbnb listings from **2019 only**
