# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Database connection setup
connection_string = st.secrets['db_url']
engine = create_engine(connection_string)

# Query function with caching
@st.cache_data
def load_data(symbol):
    query = f"""
    SELECT *
    FROM "MarketHistoricalData"
    WHERE "Symbol" = '{symbol}'
    """
    data = pd.read_sql_query(query, engine)
    data['CountDate'] = pd.to_datetime(data['CountDate'])
    data.set_index('CountDate', inplace=True)
    return data

# Predefined feature and target columns
FEATURES = ["openprice", "highprice", "lowprice", "volume"]
TARGET = "closeprice"

# App Title
st.title("Commodity Price Prediction: Lumber & Copper Futures")

# Commodity Selection
commodity_mapping = {"Lumber Futures (LBS=F)": "LBS=F", "Copper Futures (HG=F)": "HG=F"}
commodity = st.sidebar.selectbox("Select Commodity", options=list(commodity_mapping.keys()))
symbol = commodity_mapping[commodity]

# Load and display data
data = load_data(symbol)
st.write(f"Data Preview for {commodity}:")
st.dataframe(data.head())

# Filter pre-Q3 2024 data
pre_q3_2024_data = data[data.index < "2024-07-01"]
post_q3_2024_data = data[data.index >= "2024-07-01"]

# Preprocess data
st.header("Data Preprocessing")

# Handle missing data and standardize features
X_pre_q3 = pre_q3_2024_data[FEATURES]
y_pre_q3 = pre_q3_2024_data[TARGET]

# Initialize and fit the imputer for pre-Q3 data
imputer = SimpleImputer(strategy='mean')
X_pre_q3 = imputer.fit_transform(X_pre_q3)

# Standardize the features
scaler = StandardScaler()
X_pre_q3 = scaler.fit_transform(X_pre_q3)

# Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_pre_q3, y_pre_q3, test_size=0.2, random_state=42)

# Train models and select the best one
st.header("Model Training and Evaluation")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
}

model_scores = {}
best_model = None
best_r2 = float("-inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    model_scores[name] = {"MAE": mae, "R2": r2}
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# Display model results
results = pd.DataFrame(model_scores).T.sort_values(by="R2", ascending=False)
st.write("Model Performance:")
st.dataframe(results)

# Display the top 2 models based on R2
top_2_models = results.nlargest(2, "R2")
st.write("Top 2 Models Based on R2:")
st.dataframe(top_2_models)

# Create a dictionary to store predictions for the top 2 models
predictions_dict = {}

# Predict Q3-Q4 2024 data for the top 2 models
if not post_q3_2024_data.empty:
    X_post_q3 = post_q3_2024_data[FEATURES]
    y_actual_post_q3 = post_q3_2024_data[TARGET]

    # Handle missing values in post-Q3 data
    if X_post_q3.isnull().values.any():
        st.warning("NaN values detected in post-Q3 features. Imputing missing values...")
        X_post_q3 = imputer.transform(X_post_q3)
    else:
        X_post_q3 = X_post_q3.values  # Convert to numpy array

    # Standardize the features
    X_post_q3 = scaler.transform(X_post_q3)

    # Predict and store results for the top 2 models
    for model_name in top_2_models.index:
        model = models[model_name]
        predictions = model.predict(X_post_q3)
        predictions_dict[model_name] = predictions

    # Create a DataFrame to compare actual vs. predicted values
    comparison_df = pd.DataFrame({
        "Actual Close": y_actual_post_q3
    }, index=post_q3_2024_data.index)

    for model_name, predictions in predictions_dict.items():
        comparison_df[f"Predicted Close ({model_name})"] = predictions

    # Calculate the difference between actual and predicted values for Q3-Q4 2024
    for model_name in predictions_dict.keys():
        comparison_df[f"Difference ({model_name})"] = comparison_df["Actual Close"] - comparison_df[f"Predicted Close ({model_name})"]

    # Display the comparison DataFrame
    st.write("Comparison of Actual vs Predicted for Q3-Q4 2024:")
    st.dataframe(comparison_df)

    # Line chart for actual vs predicted for the top 2 models
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df["Actual Close"], mode='lines+markers', name='Actual Close'))

    for model_name in predictions_dict.keys():
        fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[f"Predicted Close ({model_name})"], mode='lines+markers', name=f'Predicted Close ({model_name})'))

    fig.update_layout(title="Actual vs Predicted Close Prices for Q3-Q4 2024 (Top 2 Models)",
                      xaxis_title="Date",
                      yaxis_title="Close Price",
                      legend_title="Legend")
    st.plotly_chart(fig)
