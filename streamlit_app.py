# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from sqlalchemy import create_engine
import xgboost as xgb

# Import secrets from Streamlit
db_url = st.secrets["db_url"]

# Title and Description
st.title('Commodity Price Prediction App')
st.markdown('''
This app predicts commodity prices using multiple machine learning models, including:
- Linear Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)

The data is retrieved from a PostgreSQL database and used to train predictive models.
''')

# Model Descriptions
st.sidebar.header('Model Descriptions')
model_descriptions = {
    'Linear Regression': 'A linear approach to modeling the relationship between a dependent variable and one or more independent variables.',
    'Random Forest': 'An ensemble learning method that builds multiple decision trees and merges them for more accurate and stable predictions.',
    'Gradient Boosting': 'An ensemble technique that builds models sequentially, each new model correcting errors made by the previous ones.',
    'SVR': 'Support Vector Regression uses a similar principle to SVM classification but is used for regression problems.',
    'KNN': 'K-Nearest Neighbors is a non-parametric method that predicts the target by finding the k closest data points in the feature space.',
    'XGBoost': 'An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.',
    'AdaBoost': 'An ensemble technique that combines multiple weak classifiers to create a strong classifier, focusing on errors of previous iterations.',
    'Decision Tree': 'A non-parametric supervised learning method that predicts the value of a target by learning decision rules inferred from features.',
    'Extra Trees': 'An ensemble learning method that aggregates results from multiple randomized decision trees.'
}
st.sidebar.subheader('Descriptions')
for model_name, description in model_descriptions.items():
    st.sidebar.markdown(f"**{model_name}**: {description}")

# Step 1: Data Acquisition
st.sidebar.header('Data Settings')
engine = create_engine(db_url)

df = pd.read_sql("SELECT * FROM historical_data_commodities_temp", con=engine)

# Add a dropdown for selecting the title from the schema
st.sidebar.subheader('Select Title')
title_options = df['Title'].unique()
selected_title = st.sidebar.selectbox('Select Title', title_options)

# Retrieve data for the selected title
df = pd.read_sql(f"SELECT * FROM historical_data_commodities_temp WHERE \"Title\" = '{selected_title}'", con=engine)

# Step 2: Data Cleaning and Processing
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# Step 3: Feature Engineering
df['Month'] = df.index.month
df['Year'] = df.index.year
df['Day'] = df.index.day
df['Quarter'] = df.index.quarter
df['5_Day_Moving_Avg'] = df['Close'].rolling(window=5).mean()
df['Close_Lag_1'] = df['Close'].shift(1)
df['Close_Lag_2'] = df['Close'].shift(2)

# Splitting Training and Test Data
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

# Display data set size
st.sidebar.write(f"Training set size: {len(train_df)} rows")
st.sidebar.write(f"Test set size: {len(test_df)} rows")

# Define features and target variable
available_features = ['Volume', 'Open', 'High', 'Low', '5_Day_Moving_Avg', 'Close_Lag_1', 'Close_Lag_2', 'Month', 'Year', 'Day', 'Quarter']
features = [feature for feature in available_features if feature in df.columns and df[feature].notna().any()]
target = 'Close'

# Ensure future_df has only available features
def get_available_features(df, features):
    return [feature for feature in features if feature in df.columns]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Handling missing values by imputing with the mean
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Model Training and Prediction
st.sidebar.subheader('Model Selection')
models = st.sidebar.multiselect('Choose Models', [
    'Linear Regression', 'Random Forest', 'Gradient Boosting', 'SVR', 'KNN', 'XGBoost', 'AdaBoost', 'Decision Tree', 'Extra Trees'
])

results = {}
if 'Linear Regression' in models:
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    results['Linear Regression'] = {
        'y_pred': y_pred_linear,
        'MAE': mean_absolute_error(y_test, y_pred_linear),
        'R2': r2_score(y_test, y_pred_linear)
    }

if 'Random Forest' in models:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = {
        'y_pred': y_pred_rf,
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'R2': r2_score(y_test, y_pred_rf)
    }

if 'Gradient Boosting' in models:
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    results['Gradient Boosting'] = {
        'y_pred': y_pred_gb,
        'MAE': mean_absolute_error(y_test, y_pred_gb),
        'R2': r2_score(y_test, y_pred_gb)
    }

if 'SVR' in models:
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)
    results['SVR'] = {
        'y_pred': y_pred_svr,
        'MAE': mean_absolute_error(y_test, y_pred_svr),
        'R2': r2_score(y_test, y_pred_svr)
    }

if 'KNN' in models:
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    results['KNN'] = {
        'y_pred': y_pred_knn,
        'MAE': mean_absolute_error(y_test, y_pred_knn),
        'R2': r2_score(y_test, y_pred_knn)
    }

if 'XGBoost' in models:
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'y_pred': y_pred_xgb,
        'MAE': mean_absolute_error(y_test, y_pred_xgb),
        'R2': r2_score(y_test, y_pred_xgb)
    }

if 'AdaBoost' in models:
    ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)
    ada_model.fit(X_train, y_train)
    y_pred_ada = ada_model.predict(X_test)
    results['AdaBoost'] = {
        'y_pred': y_pred_ada,
        'MAE': mean_absolute_error(y_test, y_pred_ada),
        'R2': r2_score(y_test, y_pred_ada)
    }

if 'Decision Tree' in models:
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    results['Decision Tree'] = {
        'y_pred': y_pred_dt,
        'MAE': mean_absolute_error(y_test, y_pred_dt),
        'R2': r2_score(y_test, y_pred_dt)
    }

if 'Extra Trees' in models:
    et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et_model.fit(X_train, y_train)
    y_pred_et = et_model.predict(X_test)
    results['Extra Trees'] = {
        'y_pred': y_pred_et,
        'MAE': mean_absolute_error(y_test, y_pred_et),
        'R2': r2_score(y_test, y_pred_et)
    }

# Determine the best model based on R2 score
best_model = max(results.items(), key=lambda x: x[1]['R2'])[0] if results else None

# Display Results
if results:
    comparison_df = pd.DataFrame({'Actual Close': y_test})
    for model_name, result in results.items():
        comparison_df[f'{model_name} Predicted Close'] = result['y_pred']
        comparison_df[f'{model_name} Difference'] = comparison_df['Actual Close'] - comparison_df[f'{model_name} Predicted Close']
        st.write(f"**{model_name}** - MAE: {result['MAE']:.2f}, R2: {result['R2']:.2f}")

    # Plot historical results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison_df.index,
        y=comparison_df['Actual Close'],
        mode='lines+markers',
        name='Actual Close'
    ))
    for model_name in results.keys():
        fig.add_trace(go.Scatter(
            x=comparison_df.index,
            y=comparison_df[f'{model_name} Predicted Close'],
            mode='lines+markers',
            name=f'{model_name} Predicted Close'
        ))

    # Add toggles for different time ranges
    date_range = st.sidebar.selectbox('Select Date Range for Analysis', ['1 Week', '3 Months', 'Max'])
    if date_range == '1 Week':
        if len(comparison_df) >= 7:
            fig.update_xaxes(range=[comparison_df.index[-7], comparison_df.index[-1]])
    elif date_range == '3 Months':
        if len(comparison_df) >= 90:
            fig.update_xaxes(range=[comparison_df.index[-90], comparison_df.index[-1]])
    elif date_range == 'Max':
        # Show all data points for the selected title
        fig.update_xaxes(range=[comparison_df.index[0], comparison_df.index[-1]])

    fig.update_layout(
        title='Actual vs Predicted Close Values with Different Models',
        xaxis_title='Date',
        yaxis_title='Close Value',
        legend_title='Legend'
    )
    st.plotly_chart(fig)

    # Display the comparison DataFrame
    st.write("### Comparison of Actual vs Predicted Values")
    st.dataframe(comparison_df)

    # Generate Dynamic Report
    report = """
    ### Dynamic Report on Prediction Accuracy
    Below is the performance comparison of different models in predicting commodity prices:
    """
    for model_name, result in results.items():
        report += f"- **{model_name}**: MAE: {result['MAE']:.2f}, R-squared: {result['R2']:.2f}\n"
    st.markdown(report)
else:
    st.write("Please select at least one model to run the predictions.")

# Step 4: Future Predictions (Next Month)
if best_model is not None:
    # Prepare future data (next 30 days)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    future_df = pd.DataFrame(index=future_dates)

    # Add the features for the future dates
    future_df['Month'] = future_df.index.month
    future_df['Year'] = future_df.index.year
    future_df['Day'] = future_df.index.day
    future_df['Quarter'] = future_df.index.quarter

    # Initialize the first values for lag and moving average features
    future_df['5_Day_Moving_Avg'] = df['Close'].iloc[-5:].mean()
    future_df['Close_Lag_1'] = df['Close'].iloc[-1]
    future_df['Close_Lag_2'] = df['Close'].iloc[-2]

    # Ensure future_df has all features used during training
    for feature in features:
        if feature not in future_df.columns:
            future_df[feature] = 0

    # Iteratively update lag features for realistic prediction
    future_predictions = []
    for i in range(len(future_df)):
        row = future_df.iloc[i].copy()
        available_features = get_available_features(future_df, features)
        if best_model == 'Linear Regression':
            prediction = linear_model.predict([row[available_features]])[0]
        elif best_model == 'Random Forest':
            prediction = rf_model.predict([row[available_features]])[0]
        elif best_model == 'Gradient Boosting':
            prediction = gb_model.predict([row[available_features]])[0]
        elif best_model == 'SVR':
            prediction = svr_model.predict([row[available_features]])[0]
        elif best_model == 'KNN':
            prediction = knn_model.predict([row[available_features]])[0]
        elif best_model == 'XGBoost':
            prediction = xgb_model.predict([row[available_features]])[0]
        elif best_model == 'AdaBoost':
            prediction = ada_model.predict([row[available_features]])[0]
        elif best_model == 'Decision Tree':
            prediction = dt_model.predict([row[available_features]])[0]
        elif best_model == 'Extra Trees':
            prediction = et_model.predict([row[available_features]])[0]

        # Update the dataframe with the predicted value
        future_predictions.append(prediction)
        if i + 1 < len(future_df):
            future_df.iloc[i + 1]['Close_Lag_1'] = prediction
            future_df.iloc[i + 1]['Close_Lag_2'] = future_df.iloc[i]['Close_Lag_1']
            future_df.iloc[i + 1]['5_Day_Moving_Avg'] = future_df['Close_Lag_1'].iloc[max(0, i - 4):i + 1].mean()

    # Ensure future_predictions has the same length as future_df before assigning
    if len(future_predictions) == len(future_df):
        future_df['Predicted_Close'] = future_predictions

    # Plot future predictions
    st.write("### Future Predictions for Next 30 Days")
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df['Predicted_Close'],
        mode='lines+markers',
        name='Predicted Close'
    ))
    fig_future.update_layout(
        title='Predicted Close Values for Next 30 Days',
        xaxis_title='Date',
        yaxis_title='Predicted Close Value',
        legend_title='Legend'
    )
    st.plotly_chart(fig_future)

    # Display the future predictions DataFrame
    st.write("### Future Predicted Values")
    st.dataframe(future_df[['Predicted_Close']])
else:
    st.write("No model was selected for predictions, so future predictions cannot be generated.")
