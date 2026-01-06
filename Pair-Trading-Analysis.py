import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Set page configuration
st.set_page_config(layout="wide")
st.title("Paired Trading Analysis")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["XGBoost Model", "LSTM Model","ARIMA Model", "Prediciton of entry/exit using LSTM"])

# Tab 1: XGBoost Model
with tab1:
    st.header("Select 2 Stocks for Paired Trading Analysis - XGBoost")
    
    # Load stock data
    df_axis = pd.read_excel('stock_data_axis.xlsx')
    stock = "Axis Bank"
    df_icici = pd.read_excel('stock_data_icici.xlsx')
    df_1 = "ICICI Bank"
    df_sbi = pd.read_excel('stock_data_sbi.xlsx')
    df_2 = "SBI Bank"
    df_bob = pd.read_excel('stock_data_bob.xlsx')
    df_3 = "Bank of Baroda"
    df_idbi = pd.read_excel('stock_data_idbi.xlsx')
    df_4 = "IDBI Bank"
    df_canara = pd.read_excel('stock_data_Canara.xlsx')
    df_5 = "Canara Bank"

    l = [stock, df_1, df_2, df_3, df_4, df_5]
    stock1 = st.selectbox("Select a Stock", l, key="xgboost_stock1")
    l.remove(stock1)
    l1 = l.copy()
    stock2 = st.selectbox("Select a Stock", l1, key="xgboost_stock2")

    # Map selection to dataframes
    stock_map = {
        "Axis Bank": df_axis,
        "ICICI Bank": df_icici,
        "SBI Bank": df_sbi,
        "Bank of Baroda": df_bob,
        "IDBI Bank": df_idbi,
        "Canara Bank": df_canara
    }
    df = stock_map[stock1]
    df1 = stock_map[stock2]

    # Convert Date columns to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df1['Date'] = pd.to_datetime(df1['Date'])

    # Date selection
    default_start = pd.to_datetime("2010-01-04")
    default_end = pd.to_datetime("2024-12-31")

    with st.expander("Customize Analysis Date Range"):
        custom_start = pd.to_datetime(st.date_input("Start Date", default_start.date(), key="xgboost_start_date"))
        custom_end = pd.to_datetime(st.date_input("End Date", default_end.date(), key="xgboost_end_date"))

    if custom_start >= custom_end:
        st.error("Invalid date range. Start date must be before end date.")
        st.stop()

    df = df[(df['Date'] >= custom_start) & (df['Date'] <= custom_end)].reset_index(drop=True)
    df1 = df1[(df1['Date'] >= custom_start) & (df1['Date'] <= custom_end)].reset_index(drop=True)
    st.success(f"Using data from {custom_start.date()} to {custom_end.date()}")
    st.write(f"Number of trading days: **{len(df)} days**")
    st.header("XGBoost Model:")
    st.markdown("""
    We are using **XGBoost**, a powerful machine learning model, to predict trading signals for a paired trading strategy.
                """)

    # Adding a button to display detailed information about the XGBoost model
    if st.button("Click here to learn more about the XGBoost Model"):
        st.markdown("""
        ### What are we trying to achieve with XGBoost?

        1. **Model Objective**: Our goal is to train an **XGBoost classifier** to predict whether we should **Buy**, **Hold**, or **Sell** a stock based on the price spread between two stocks.

        2. **Features**: We use technical indicators like **RSI**, **Bollinger Bands**, and **MACD**, along with the **lagged spread**, to create features that represent historical stock data.

        3. **Target**: The model's target is to classify the price spread into three categories:
           - **Buy**: When the spread suggests a good buying opportunity.
           - **Hold**: When the spread is neutral or within a normal range.
           - **Sell**: When the spread indicates an opportunity to sell.
        """)

    st.divider()
    # Merge DataFrames
    df_merged = pd.merge(df[['Date', 'Close']], df1[['Date', 'Close']], on='Date', suffixes=(f'_{stock1}', f'_{stock2}'))
    df_merged.dropna(inplace=True)

    # Feature Engineering
    df_merged['Spread'] = df_merged[f'Close_{stock1}'] - df_merged[f'Close_{stock2}']
    df_merged['Lag1'] = df_merged['Spread'].shift(1)
    df_merged['Lag2'] = df_merged['Spread'].shift(2)

    # RSI (Relative Strength Index)
    delta = df_merged['Spread'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df_merged['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df_merged['Spread'].rolling(window=20).mean()
    rolling_std = df_merged['Spread'].rolling(window=20).std()
    df_merged['BB_upper'] = rolling_mean + (rolling_std * 2)
    df_merged['BB_lower'] = rolling_mean - (rolling_std * 2)

    # MACD
    ema_12 = df_merged['Spread'].ewm(span=12, adjust=False).mean()
    ema_26 = df_merged['Spread'].ewm(span=26, adjust=False).mean()
    df_merged['MACD'] = ema_12 - ema_26

    # Drop NA and define features
    df_merged.dropna(inplace=True)
    features = ['Lag1', 'Lag2', 'RSI', 'BB_upper', 'BB_lower', 'MACD']
    X = df_merged[features]

    # Classify the Spread into Buy/Hold/Sell signals
    spread_mean = df_merged['Spread'].mean()
    spread_std = df_merged['Spread'].std()

    # Optional: Making thresholds adjustable
    buy_threshold_slider = st.slider("Buy Threshold Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    sell_threshold_slider = st.slider("Sell Threshold Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    # Update thresholds based on slider input
    buy_threshold = spread_mean - spread_std * buy_threshold_slider
    sell_threshold = spread_mean + spread_std * sell_threshold_slider

    def classify_signal(spread):
        if spread > sell_threshold:
            return 2  # Sell
        elif spread < buy_threshold:
            return 0  # Buy
        else:
            return 1  # Hold

    # Apply classification
    df_merged['Signal'] = df_merged['Spread'].apply(classify_signal)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, df_merged['Signal'], shuffle=False, test_size=0.2)

    # Hyperparameter Grid
    params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 1],
        'colsample_bytree': [0.7, 1]
    }

    # GridSearch
    xgb = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    st.divider()
    # Header for Actual Signal Plot
    st.header("XGBoost - Actual Signals Plot")
    # Plot Buy/Hold/Sell signals - Actual only
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_merged['Date'].iloc[-len(y_test):], y_test, label="Actual Signal", marker='o')
    ax.set_title("XGBoost - Actual Signals (Buy/Hold/Sell)")
    ax.set_xlabel('Date')
    ax.set_ylabel('Signal')
    ax.legend()
    st.pyplot(fig)

    # Inference for actual signals
    st.info("Actual Signal Plot Insight:\nThis plot shows the true Buy (+1), Hold (0), and Sell (-1) signals generated during the test period. It helps to visually analyze the timing and frequency of trade opportunities as identified by the model.")
    st.divider()

    # Header for Actual vs Predicted Comparison
    st.header("XGBoost - Actual vs Predicted Signals Comparison")
    # Actual vs Predicted Signals Plot
    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
    ax_pred.plot(df_merged['Date'].iloc[-len(y_test):], y_test, label="Actual Signal", marker='o')
    ax_pred.plot(df_merged['Date'].iloc[-len(y_test):], y_pred, label="Predicted Signal", linestyle='--', marker='x')
    ax_pred.set_title("Actual vs Predicted Signals (Buy/Hold/Sell)")
    ax_pred.set_xlabel('Date')
    ax_pred.set_ylabel('Signal')
    ax_pred.legend()
    st.pyplot(fig_pred)

    # Inference for actual vs predicted comparison
    st.success("Model Prediction Performance:\nThis comparison shows how closely the XGBoost model's predicted signals align with the actual ones. Matching points suggest good classification accuracy, while divergence may indicate areas where the model misclassifies signals, potentially affecting trading performance.")
    st.divider()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])  # [Buy, Hold, Sell]

    # Plotting
    st.subheader("XGBoost Performance:")
    st.write(f"**Accuracy Score:** {accuracy * 100:.2f}%")

    # Inference on accuracy
    if accuracy >= 0.80:
        st.success("Excellent model performance: The XGBoost classifier is making highly accurate predictions on the test set, which indicates it has captured the patterns in the signal data effectively.")
    elif 0.60 <= accuracy < 0.80:
        st.info("Moderate model performance: The model is reasonably accurate, but there may be room for improvement through additional feature engineering or hyperparameter tuning.")
    else:
        st.warning("Low model accuracy: The predictions may not be reliable. Consider revisiting feature selection, signal labeling, or model tuning.")

    st.divider()

    # Classification Report with explicit labels for all 3 classes
    st.subheader("Classification Report")

    # Generate report
    report = classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Sell'], labels=[0, 1, 2])
    st.code(report)

    # Inference
    st.warning(
        "Interpretation:\n"
        "- **Precision** indicates how many of the predicted signals were correct.\n"
        "- **Recall** shows how well the model captured all actual instances of each signal.\n"
        "- **F1-score** balances both precision and recall.\n\n"
        "Pay close attention to the 'Buy' and 'Sell' classes. If their precision or recall is low, the model might generate false signals, potentially impacting trading decisions. High performance on the 'Hold' class alone may not be sufficient if Buy/Sell accuracy is weak."
    )
    st.divider()


    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Buy', 'Hold', 'Sell'], yticklabels=['Buy', 'Hold', 'Sell'])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix - Spread Signal Classification")
    st.pyplot(fig_cm)
    st.divider()
    st.subheader("Regression Error Metrics")
    # Regression Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    # Create two columns
    col1, col2 = st.columns(2)

    # Display regression metrics in the columns
    with col1:
        st.markdown("**Mean Squared Error (MSE):**")
        st.markdown("**Root Mean Squared Error (RMSE):**")
        st.markdown("**Mean Absolute Error (MAE):**")
        st.markdown("**R² (Coefficient of Determination):**")

    with col2:
        st.markdown(f"{mse:.4f}")
        st.markdown(f"{rmse:.4f}")
        st.markdown(f"{mae:.4f}")
        st.markdown(f"{r2:.4f}")

    # Inference on RMSE (Root Mean Squared Error)
    st.info(
        f"**Inference on RMSE:**\n"
        "- A **low RMSE** indicates that the model's predictions are close to the actual values, suggesting a good fit.\n"
        "- A **high RMSE** means that the predictions are off by a larger margin, and the model might need further tuning."
    )

    # Threshold-based Inference for R²
    threshold_r2 = 0.7  # You can adjust this threshold based on your requirements
    if r2 >= threshold_r2:
        st.success(f"**Good Model Fit:**\nThe model's R² score of {r2:.4f} suggests that it explains a high percentage of the variance in the data, indicating a strong fit.")
    elif 0.4 <= r2 < threshold_r2:
        st.warning(f"**Moderate Model Fit:**\nThe model's R² score of {r2:.4f} suggests a moderate level of fit. You may want to consider adding more features or fine-tuning the model.")
    else:
        st.error(f"**Poor Model Fit:**\nThe R² score of {r2:.4f} indicates that the model is not explaining much of the variance in the data. Consider revisiting the feature engineering process or trying a different model.")
 
    # Threshold-based Inference for MAE
    threshold_mae = 0.05  # Adjust the threshold based on your scale of prediction
    if mae <= threshold_mae:
        st.success(f"**Low MAE:**\nThe model has an MAE of {mae:.4f}, indicating that on average, the predictions are off by a small amount, showing good performance.")
    elif 0.05 < mae <= 0.1:
        st.warning(f"**Moderate MAE:**\nThe MAE of {mae:.4f} suggests that there is room for improvement in model accuracy. Consider improving feature selection or hyperparameters.")
    else:
        st.error(f"**High MAE:**\nThe model's MAE of {mae:.4f} suggests significant errors in the predictions. This indicates a need for further model optimization or different algorithms.")

    # Threshold-based Inference for MSE
    threshold_mse = 0.01  # Set your threshold based on prediction scale
    if mse <= threshold_mse:
        st.success(f"**Low MSE:**\nThe model has an MSE of {mse:.4f}, indicating good predictive accuracy with small error values.")
    elif 0.01 < mse <= 0.05:
        st.warning(f"**Moderate MSE:**\nThe MSE of {mse:.4f} suggests the model's predictions could be more accurate. Fine-tuning the model may help.")
    else:
        st.error(f"**High MSE:**\nThe model's MSE of {mse:.4f} indicates significant errors in prediction, suggesting a poor fit or the need for model improvement.")

# Tab 2: LSTM Model
with tab2:
    st.header("Select 2 Stocks for Paired Trading Analysis - LSTM")

    # Load stock data
    df_axis = pd.read_excel('stock_data_axis.xlsx')
    stock = "Axis Bank"
    df_icici = pd.read_excel('stock_data_icici.xlsx')
    df_1 = "ICICI Bank"
    df_sbi = pd.read_excel('stock_data_sbi.xlsx')
    df_2 = "SBI Bank"
    df_bob = pd.read_excel('stock_data_bob.xlsx')
    df_3 = "Bank of Baroda"
    df_idbi = pd.read_excel('stock_data_idbi.xlsx')
    df_4 = "IDBI Bank"
    df_canara = pd.read_excel('stock_data_Canara.xlsx')
    df_5 = "Canara Bank"

    l = [stock, df_1, df_2, df_3, df_4, df_5]
    stock1 = st.selectbox("Select a Stock", l, key="lstm_stock1")
    l.remove(stock1)
    stock2 = st.selectbox("Select another Stock", l, key="lstm_stock2")

    # Map selection to dataframes
    stock_map = {
        "Axis Bank": df_axis,
        "ICICI Bank": df_icici,
        "SBI Bank": df_sbi,
        "Bank of Baroda": df_bob,
        "IDBI Bank": df_idbi,
        "Canara Bank": df_canara
    }
    df = stock_map[stock1]
    df1 = stock_map[stock2]

    # Convert date columns
    df['Date'] = pd.to_datetime(df['Date'])
    df1['Date'] = pd.to_datetime(df1['Date'])

    # Date selection
    default_start = pd.to_datetime("2010-01-04")
    default_end = pd.to_datetime("2024-12-31")

    with st.expander("Customize Analysis Date Range"):
        custom_start = pd.to_datetime(st.date_input("Start Date", default_start.date(), key="lstm_start_date"))
        custom_end = pd.to_datetime(st.date_input("End Date", default_end.date(), key="lstm_end_date"))

    if custom_start >= custom_end:
        st.error("Invalid date range. Start date must be before end date.")
        st.stop()

    df = df[(df['Date'] >= custom_start) & (df['Date'] <= custom_end)].reset_index(drop=True)
    df1 = df1[(df1['Date'] >= custom_start) & (df1['Date'] <= custom_end)].reset_index(drop=True)
    st.success(f"Using data from {custom_start.date()} to {custom_end.date()}")
    st.write(f"Number of trading days: **{len(df)} days**")

    # LSTM Model Implementation
    st.header("LSTM Modeling")

    # Merge selected stocks' prices
    merged_df = pd.merge(df[['Date', 'Close']], df1[['Date', 'Close']], on='Date', suffixes=(f'_{stock1}', f'_{stock2}'))

    # Create spread and z-score
    merged_df['Spread'] = np.log(merged_df[f'Close_{stock1}']) - np.log(merged_df[f'Close_{stock2}'])
    merged_df['Z_Score'] = (merged_df['Spread'] - merged_df['Spread'].mean()) / merged_df['Spread'].std()

    # Clean column names to remove any leading/trailing spaces
    merged_df.columns = merged_df.columns.str.strip()

    # Create lagged features
    merged_df['Z_Lag1'] = merged_df['Z_Score'].shift(1)
    merged_df['Z_Lag2'] = merged_df['Z_Score'].shift(2)

    # Define signal with more robust logic
    def generate_signal(z):
        if z > 1:
            return 1  # Sell
        elif z < -1:
            return -1  # Buy
        else:
            return 0  # Hold

    merged_df['Signal'] = merged_df['Z_Score'].apply(generate_signal)
    merged_df.dropna(inplace=True)

    # Encode signals for classification
    signal_map = {-1: 0, 0: 1, 1: 2}
    merged_df['Signal_Mapped'] = merged_df['Signal'].map(signal_map)

    # Prepare data for LSTM
    X = merged_df[['Z_Lag1', 'Z_Lag2']].values
    y = merged_df['Signal_Mapped'].values

    # Reshape input to be 3D [samples, time steps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    # LSTM model with reduced complexity
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))  # Fewer units
    model.add(Dropout(0.5))  # Increased dropout rate
    model.add(Dense(3, activation='softmax'))  # 3 classes for signals

    # Compile model with a smaller learning rate
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit model with fewer epochs
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)  # Fewer epochs and larger batch size

    # Predict
    y_pred_mapped = model.predict(X_test)
    y_pred = np.argmax(y_pred_mapped, axis=1)

    # Update merged_df with predictions
    merged_df['Prediction'] = np.nan
    merged_df.iloc[-len(y_test):, merged_df.columns.get_loc('Prediction')] = y_pred

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Display metrics
    st.subheader("📏 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(" Accuracy", f"{accuracy:.2%}")
    col2.metric("F1 Score (Macro)", f"{f1:.4f}")
    col3.metric(" Precision", f"{precision:.4f}")
    col4.metric(" Recall", f"{recall:.4f}")

    st.subheader("Inference Based on Metrics")

    # Define thresholds for quality
    high_threshold = 0.85
    medium_threshold = 0.65

    # Accuracy
    if accuracy >= high_threshold:
        st.success(f"Excellent Accuracy: {accuracy:.2%} — The model is highly reliable in predicting trading signals.")
    elif accuracy >= medium_threshold:
        st.warning(f"Moderate Accuracy: {accuracy:.2%} — Model is decent, but there’s room for improvement.")
    else:
        st.error(f"Low Accuracy: {accuracy:.2%} — Predictions may be unreliable. Consider feature tuning or using more data.")

    # F1 Score
    if f1 >= high_threshold:
        st.success(f"Strong F1 Score: {f1:.4f} — The model balances precision and recall effectively across classes.")
    elif f1 >= medium_threshold:
        st.warning(f"Average F1 Score: {f1:.4f} — Balance between precision and recall is moderate.")
    else:
        st.error(f"Poor F1 Score: {f1:.4f} — Model struggles with one or more signal classes.")

    # Precision
    if precision >= high_threshold:
        st.success(f"High Precision: {precision:.4f} — Model is confident and mostly correct in its predictions.")
    elif precision >= medium_threshold:
        st.warning(f"Medium Precision: {precision:.4f} — Some predictions are off. Model could be refined.")
    else:
        st.error(f" Low Precision: {precision:.4f} — Too many false predictions. Check class imbalance or feature relevance.")

    # Recall
    if recall >= high_threshold:
        st.success(f"High Recall: {recall:.4f} — Model is successfully identifying most true signals.")
    elif recall >= medium_threshold:
        st.warning(f"Moderate Recall: {recall:.4f} — Model misses some true signals.")
    else:
        st.error(f"Low Recall: {recall:.4f} — Model fails to detect many actual signals. Consider improving sensitivity.")

    st.subheader("Z-Score with Predicted Trading Signals")

    # Predicted signals graph
    fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
    ax_pred.plot(merged_df['Date'], merged_df['Z_Score'], label='Z-Score', color='black', alpha=0.7)
    ax_pred.axhline(1, color='red', linestyle='--', linewidth=1, label='Sell Threshold (1)')
    ax_pred.axhline(-1, color='green', linestyle='--', linewidth=1, label='Buy Threshold (-1)')

    # Filter only test set for predictions
    buy_signals_pred = merged_df[merged_df['Prediction'] == 0]  # Buy
    sell_signals_pred = merged_df[merged_df['Prediction'] == 2]  # Sell

    # Plot predicted buy/sell signals
    ax_pred.scatter(buy_signals_pred['Date'], buy_signals_pred['Z_Score'], color='green', label='Predicted Buy', marker='^', s=100)
    ax_pred.scatter(sell_signals_pred['Date'], sell_signals_pred['Z_Score'], color='red', label='Predicted Sell', marker='v', s=100)

    ax_pred.set_title("Z-Score Over Time with Predicted Trading Signals")
    ax_pred.legend()
    st.pyplot(fig_pred)

    st.subheader("Z-Score with Actual Trading Signals")

    # Actual signals graph
    fig_actual, ax_actual = plt.subplots(figsize=(12, 5))
    ax_actual.plot(merged_df['Date'], merged_df['Z_Score'], label='Z-Score', color='black', alpha=0.7)
    ax_actual.axhline(1, color='red', linestyle='--', linewidth=1, label='Sell Threshold (1)')
    ax_actual.axhline(-1, color='green', linestyle='--', linewidth=1, label='Buy Threshold (-1)')

    # Filter only test set for actual signals
    actual_buy_signals = merged_df[merged_df['Signal'] == -1]  # Actual Buy
    actual_sell_signals = merged_df[merged_df['Signal'] == 1]  # Actual Sell

    # Plot actual buy/sell signals
    ax_actual.scatter(actual_buy_signals['Date'], actual_buy_signals['Z_Score'], color='lightgreen', label='Actual Buy', marker='^', s=100, edgecolor='black')
    ax_actual.scatter(actual_sell_signals['Date'], actual_sell_signals['Z_Score'], color='salmon', label='Actual Sell', marker='v', s=100, edgecolor='black')

    ax_actual.set_title("Z-Score Over Time with Actual Trading Signals")
    ax_actual.legend()
    st.pyplot(fig_actual)

# Tab 3: Arima Model
with tab3:
    warnings.filterwarnings("ignore")
    st.title(f"Select 2 Stock for Paired Trading Analysis")

# Load stock data
    df_dict = {
        "Axis Bank": pd.read_excel('stock_data_axis.xlsx'),
        "ICICI Bank": pd.read_excel('stock_data_icici.xlsx'),
        "SBI Bank": pd.read_excel('stock_data_sbi.xlsx'),
        "Bank of Baroda": pd.read_excel('stock_data_bob.xlsx'),
        "IDBI Bank": pd.read_excel('stock_data_idbi.xlsx'),
        "Canara Bank": pd.read_excel('stock_data_Canara.xlsx'),
    }

    stock_names = list(df_dict.keys())
    stock1 = st.selectbox("Select Stock 1", stock_names)
    stock2 = st.selectbox("Select Stock 2", [s for s in stock_names if s != stock1])

    df1 = df_dict[stock1].copy()
    df2 = df_dict[stock2].copy()

    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    df1.set_index('Date', inplace=True)
    df2.set_index('Date', inplace=True)

    # Filter dates
    # Date selection
    default_start = pd.to_datetime("2010-01-04")
    default_end = pd.to_datetime("2024-12-31")

    with st.expander("Customize Analysis Date Range"):
        custom_start = pd.to_datetime(st.date_input("Start Date", default_start.date(), key="arima_start_date"))
        custom_end = pd.to_datetime(st.date_input("End Date", default_end.date(), key="arima_end_date"))

        if custom_start >= custom_end:
            st.error("Invalid date range")
            st.stop()

        df1 = df1[(df1.index >= custom_start) & (df1.index <= custom_end)]
        df2 = df2[(df2.index >= custom_start) & (df2.index <= custom_end)]
        st.info(f"Using data from {custom_start.date()} to {custom_end.date()}")
    st.header("Sarima/Arima Model:")
    st.markdown("""
    This model analyzes the spread between two selected bank stocks, forecasting future spread values using a **SARIMA (Seasonal ARIMA)** model. The model predicts the difference in closing prices (spread) between the two stocks, helping to generate trading signals (Buy, Hold, Sell) for pair trading strategies based on the forecasted spread behavior.
    """)

    # Compute spread
    spread = df1['Close'] - df2['Close']

    # ADF test
    st.subheader("ADF Test for Stationarity of Spread")
    adf_result = adfuller(spread.dropna())
    p_value = adf_result[1]
    st.write(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {p_value:.4f}")

    if p_value > 0.05:
        st.warning("The spread is non-stationary. We'll apply first differencing.")
        spread = spread.diff().dropna()
    else:
        st.success("The spread is stationary. No differencing required.")

    # Inference
    st.info("""
    **Inference:**  
    A stationary spread is essential for meaningful time series modeling. Since stationarity was confirmed (or achieved through differencing), we proceed with SARIMA modeling.
    """)
    st.divider()
    # Train-test split
    train_size = int(0.8 * len(spread))
    train, test = spread[:train_size], spread[train_size:]

    # SARIMA Grid Search
    st.subheader("SARIMA Hyperparameter Tuning (AIC-Based Selection)")
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    s = 12

    best_aic = float("inf")
    best_order = None
    best_seasonal = None
    best_model = None

    for order in itertools.product(p, d, q):
        for seasonal in itertools.product(P, D, Q):
            try:
                model = SARIMAX(train,
                                order=order,
                                seasonal_order=(seasonal[0], seasonal[1], seasonal[2], s),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_seasonal = seasonal
                    best_model = results
            except:
                continue

    st.success(f"Best SARIMA Order: {best_order}, Seasonal Order: {best_seasonal}, Seasonal Period: {s} | AIC: {best_aic:.2f}")

    st.info("""
    **Inference:**  
    SARIMA hyperparameter tuning selected the model with the lowest AIC score, which indicates a good trade-off between model fit and complexity.
    """)
    st.divider()
    st.subheader("Actual and Forecasted Spread")
    # Forecast
    forecast = best_model.get_forecast(steps=len(test)).predicted_mean
    forecast.index = test.index

    # Classification helper
    def classify(val, lower=-1, upper=1):
        if val < lower:
            return -1  # Buy
        elif val > upper:
            return 1   # Sell
        else:
            return 0   # Hold

    actual_signals = test.apply(classify)
    predicted_signals = forecast.apply(classify)
    # Plot showing Actual and Forecasted Spread
    fig, ax = plt.subplots(figsize=(14, 7))
    train.plot(label="Train", ax=ax, color="black")
    test.plot(label="Test", ax=ax, color="blue")
    forecast.plot(label="Forecast", ax=ax, color="red")

    ax.set_title("Actual and Forecasted Spread")
    ax.legend()
    st.pyplot(fig)

    st.info("""
    ### **Inference on Forecasted Spread:**
    The plot above shows the actual and forecasted values of the spread. The model's forecasted spread is essential to determine whether the action is to **Buy**, **Hold**, or **Sell** based on the signal thresholds defined earlier. Here's what the forecast implies:
    - The **forecast line** helps visualize how well the model predicts the future spread.
    """)
    st.divider()
    st.subheader("Buy, Sell, Hold Signals in the forecasted region")
    # Combined Plot for Buy, Sell, Hold Signals without the Forecast Line
    fig, ax = plt.subplots(figsize=(14, 7))
    train.plot(label="Train", ax=ax, color="black")
    test.plot(label="Test", ax=ax, color="blue")

    # Plot Buy, Sell, Hold signals
    buy_signals = forecast[actual_signals == -1]
    sell_signals = forecast[actual_signals == 1]
    hold_signals = forecast[actual_signals == 0]

    ax.scatter(buy_signals.index, buy_signals, color="green", label="Buy Signal", marker="^", s=100)
    ax.scatter(sell_signals.index, sell_signals, color="red", label="Sell Signal", marker="v", s=100)
    ax.scatter(hold_signals.index, hold_signals, color="yellow", label="Hold Signal", marker="o", s=100)

    ax.set_title("Buy, Sell, Hold Signals vs Actual Spread")
    ax.legend()
    st.pyplot(fig)

    st.info("""
    ### **Inference on Buy, Sell, Hold Signals:**
    This plot illustrates the **Buy**, **Sell**, and **Hold** signals along with the actual spread values. The forecasted spread line has been removed to focus on the actions taken based on the actual spread:
    - **Green (Buy)**: The **Buy** signal is triggered when the spread falls below the lower threshold, indicating that the spread is likely to widen, making it an ideal opportunity for entering a long position.
    - **Red (Sell)**: The **Sell** signal is triggered when the spread rises above the upper threshold, indicating that the spread is likely to narrow, suggesting a short position in the second stock.
    - **Yellow (Hold)**: The **Hold** signal appears when the spread stays within the threshold range, meaning no significant movement is expected, and maintaining the current position is recommended.
    """)
    st.divider()
    st.subheader("Regression Error Metrics")

    # Metrics
    accuracy = accuracy_score(actual_signals, predicted_signals)
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)

    st.subheader("Forecast Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal Accuracy", f"{accuracy:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")
    col3.metric("MAE", f"{mae:.4f}")
    col4.metric("RMSE", f"{rmse:.4f}")

    # Inference on RMSE (Root Mean Squared Error)
    st.info(
        f"**Inference on RMSE:**\n"
        "- A **low RMSE** indicates that the model's predictions are close to the actual values, suggesting a good fit.\n"
        "- A **high RMSE** means that the predictions are off by a larger margin, and the model might need further tuning."
    )

    # Threshold-based Inference for R²
    threshold_r2 = 0.7  # You can adjust this threshold based on your requirements
    if r2 >= threshold_r2:
        st.success(f"**Good Model Fit:**\nThe model's R² score of {r2:.4f} suggests that it explains a high percentage of the variance in the data, indicating a strong fit.")
    elif 0.4 <= r2 < threshold_r2:
        st.warning(f"**Moderate Model Fit:**\nThe model's R² score of {r2:.4f} suggests a moderate level of fit. You may want to consider adding more features or fine-tuning the model.")
    else:
        st.error(f"**Poor Model Fit:**\nThe R² score of {r2:.4f} indicates that the model is not explaining much of the variance in the data. Consider revisiting the feature engineering process or trying a different model.")

    # Threshold-based Inference for MAE
    threshold_mae = 0.05  # Adjust the threshold based on your scale of prediction
    if mae <= threshold_mae:
        st.success(f"**Low MAE:**\nThe model has an MAE of {mae:.4f}, indicating that on average, the predictions are off by a small amount, showing good performance.")
    elif 0.05 < mae <= 0.1:
        st.warning(f"**Moderate MAE:**\nThe MAE of {mae:.4f} suggests that there is room for improvement in model accuracy. Consider improving feature selection or hyperparameters.")
    else:
        st.error(f"**High MAE:**\nThe model's MAE of {mae:.4f} suggests significant errors in the predictions. This indicates a need for further model optimization or different algorithms.")
    st.divider()

    st.subheader("Confusion Matrix:")
    # Generate confusion matrix
    conf_matrix = confusion_matrix(actual_signals, predicted_signals)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Buy", "Hold", "Sell"], yticklabels=["Buy", "Hold", "Sell"], ax=ax)
    ax.set_title("Confusion Matrix for Buy, Hold, Sell Signals")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Display the plot in Streamlit
    st.pyplot(fig)
    st.divider()
    st.subheader("Classification Report")

    class_report = classification_report(actual_signals, predicted_signals, target_names=["Buy", "Hold", "Sell"])

    # Display the classification report in Streamlit
    st.code(class_report)

    # Inference
    st.warning(
        "Interpretation:\n"
        "- **Precision** indicates how many of the predicted signals were correct.\n"
        "- **Recall** shows how well the model captured all actual instances of each signal.\n"
        "- **F1-score** balances both precision and recall.\n\n"
        "Pay close attention to the 'Buy' and 'Sell' classes. If their precision or recall is low, the model might generate false signals, potentially impacting trading decisions. High performance on the 'Hold' class alone may not be sufficient if Buy/Sell accuracy is weak."
    )
    st.divider()

# Tab 4: Z-Score Entry/Exit Signals
with tab4:
    st.header("Pair Trading Strategy - Z-Score Entry/Exit Signals")

    # Available stock data files
    data_files = {
        "Axis Bank": "stock_data_axis.xlsx",
        "ICICI Bank": "stock_data_icici.xlsx",
        "SBI Bank": "stock_data_sbi.xlsx",
        "Bank of Baroda": "stock_data_bob.xlsx",
        "IDBI Bank": "stock_data_idbi.xlsx",
        "Canara Bank": "stock_data_Canara.xlsx"
    }

    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df[['Date', 'Close']]

    # Select two stocks
    stocks = list(data_files.keys())
    stock1 = st.selectbox("Select First Stock", stocks, index=0)
    stock2 = st.selectbox("Select Second Stock", stocks, index=1)

    if stock1 == stock2:
        st.warning("Please select two different stocks.")
        st.stop()

    df1 = load_data(data_files[stock1])
    df2 = load_data(data_files[stock2])

    # Merge data
    df = pd.merge(df1, df2, on='Date', how='inner', suffixes=(f'_{stock1}', f'_{stock2}'))
    df.sort_values('Date', inplace=True)

    # Date filter
    default_start = df['Date'].min()
    default_end = df['Date'].max()
    with st.expander("Customize Date Range"):
        start_date = pd.to_datetime(st.date_input("Start Date", default_start.date()))
        end_date = pd.to_datetime(st.date_input("End Date", default_end.date()))

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)

    # Compute spread
    df['Spread'] = df[f'Close_{stock1}'] - df[f'Close_{stock2}']

    # Rolling stats and z-score
    window = st.slider("Z-Score Rolling Window", min_value=10, max_value=100, value=30, step=5)
    df['Spread_Mean'] = df['Spread'].rolling(window=window).mean()
    df['Spread_Std'] = df['Spread'].rolling(window=window).std()
    df['Z_Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']

    # Generate signals
    def generate_signals(z):
        if z <= -1:
            return "Buy A / Short B"
        elif z >= 1:
            return "Sell A / Long B"
        elif -0.5 < z < 0.5:
            return "Exit"
        else:
            return "Hold"

    df['Signal'] = df['Z_Score'].apply(generate_signals)

    # Plot spread and z-score
    st.subheader(" Spread & Z-Score")
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Spread plot
    ax[0].plot(df['Date'], df['Spread'], label='Spread', color='purple')
    ax[0].axhline(df['Spread'].mean(), color='gray', linestyle='--', label='Mean')
    ax[0].set_ylabel("Spread")
    ax[0].legend()
    ax[0].set_title(f"Spread: {stock1} - {stock2}")

    # Z-score plot
    ax[1].plot(df['Date'], df['Z_Score'], label='Z-Score', color='blue')
    ax[1].axhline(0, linestyle='--', color='black')
    ax[1].axhline(1, linestyle='--', color='red', label='Sell Threshold (+1)')
    ax[1].axhline(-1, linestyle='--', color='green', label='Buy Threshold (-1)')
    ax[1].fill_between(df['Date'], -0.5, 0.5, color='yellow', alpha=0.1, label='Exit Zone')
    ax[1].set_ylabel("Z-Score")
    ax[1].legend()

    st.pyplot(fig)

    # Signal preview
    st.subheader("Entry/Exit Signals")
    st.dataframe(df[['Date', f'Close_{stock1}', f'Close_{stock2}', 'Spread', 'Z_Score', 'Signal']].dropna().tail(20))

    # Signal stats
    st.subheader("Signal Statistics")
    signal_counts = df['Signal'].value_counts().reset_index()
    signal_counts.columns = ['Signal', 'Count']
    st.dataframe(signal_counts)

    # Load data for selected stocks
    df1 = pd.read_excel(data_files[stock1])
    df2 = pd.read_excel(data_files[stock2])
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])

    # Filter by selected date range
    df1 = df1[(df1['Date'] >= start_date) & (df1['Date'] <= end_date)]
    df2 = df2[(df2['Date'] >= start_date) & (df2['Date'] <= end_date)]

    # Merge and calculate spread/z-score
    merged_df = pd.merge(df1[['Date', 'Close']], df2[['Date', 'Close']], on='Date', suffixes=(f'_{stock1}', f'_{stock2}'))
    merged_df['Spread'] = np.log(merged_df[f'Close_{stock1}']) - np.log(merged_df[f'Close_{stock2}'])
    merged_df['Z_Score'] = (merged_df['Spread'] - merged_df['Spread'].mean()) / merged_df['Spread'].std()


    st.header("Predicting Entry/Exit Points (LSTM Forecasting)")

    # Prepare the data for LSTM prediction
    forecast_df = merged_df[['Z_Score']].copy()
    forecast_df['Z_Score_Lag1'] = forecast_df['Z_Score'].shift(1)
    forecast_df['Z_Score_Lag2'] = forecast_df['Z_Score'].shift(2)
    forecast_df.dropna(inplace=True)

    # Features and target
    X = forecast_df[['Z_Score_Lag1', 'Z_Score_Lag2']].values
    y = forecast_df['Z_Score'].values  # Predict next z-score

    # Reshape for LSTM
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(forecast_df[['Z_Score_Lag1', 'Z_Score_Lag2']])

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    # LSTM model
    model_pred = Sequential()
    model_pred.add(LSTM(20, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_pred.add(Dropout(0.3))
    model_pred.add(Dense(1))  # Regression output
    model_pred.compile(optimizer='adam', loss='mse')
    model_pred.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predict future z-scores
    predicted_z = model_pred.predict(X_test).flatten()

    # Decision logic: classify predicted z-score into signals
    st.sidebar.subheader("Signal Threshold Settings")
    buy_threshold = st.sidebar.slider("Buy Threshold (Entry)", min_value=-3.0, max_value=0.0, value=-1.0, step=0.1)
    sell_threshold = st.sidebar.slider("Sell Threshold (Exit)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

    def predict_signal(z):
        if z > sell_threshold:
            return "Exit (Sell)"
        elif z < buy_threshold:
            return "Entry (Buy)"
        else:
            return "Hold"


    predicted_signals = [predict_signal(z) for z in predicted_z]

    # Create result DataFrame
    result_df = merged_df.iloc[-len(y_test):].copy()
    result_df['Predicted_Z_Score'] = predicted_z
    result_df['Predicted_Action'] = predicted_signals
    signal_summary = pd.Series(predicted_signals).value_counts()
    st.write("Predicted Signal Distribution:")
    st.dataframe(signal_summary)


    # Display prediction table
    st.subheader("Predicted Entry/Exit Signals")
    st.dataframe(result_df[['Date', f'Close_{stock1}', f'Close_{stock2}', 'Predicted_Z_Score', 'Predicted_Action']].reset_index(drop=True))

    # Plot predicted signals
    st.subheader("Predicted Signals on Z-Score Chart")

    fig_pred_signal, ax_pred_signal = plt.subplots(figsize=(12, 5))
    ax_pred_signal.plot(result_df['Date'], result_df['Predicted_Z_Score'], label='Predicted Z-Score', color='black')
    ax_pred_signal.axhline(1, color='red', linestyle='--', label='Sell Threshold (1)')
    ax_pred_signal.axhline(-1, color='green', linestyle='--', label='Buy Threshold (-1)')

    # Plot actions
    for idx, row in result_df.iterrows():
        if row['Predicted_Action'] == 'Entry (Buy)':
            ax_pred_signal.scatter(row['Date'], row['Predicted_Z_Score'], color='green', marker='^', s=100, label='Predicted Entry' if idx == 0 else "")
        elif row['Predicted_Action'] == 'Exit (Sell)':
            ax_pred_signal.scatter(row['Date'], row['Predicted_Z_Score'], color='red', marker='v', s=100, label='Predicted Exit' if idx == 0 else "")

    ax_pred_signal.set_title("Predicted Z-Score with Entry/Exit Points")
    ax_pred_signal.legend()
    fig_pred_signal.tight_layout()
    st.pyplot(fig_pred_signal)

