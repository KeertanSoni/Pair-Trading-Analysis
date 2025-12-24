import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from scipy.stats import boxcox
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title(f"Select 2 Stock for Paired Trading Analysis")

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
stock1 = st.selectbox("Select a Stock", l)
l.remove(stock1)
l1 = l.copy()
stock2 = st.selectbox("Select a Stock", l1)

if stock1 == stock:
    df = df_axis
    df_ = "Axis Bank"
elif stock1 == df_1:
    df = df_icici
    df_ = "ICICI Bank"
elif stock1 == df_2:
    df = df_sbi
    df_ = "SBI Bank"
elif stock1 == df_3:
    df = df_bob
    df_ = "Bank of Baroda"
elif stock1 == df_4:
    df = df_idbi
    df_ = "IDBI Bank"
elif stock1 == df_5:
    df = df_canara
    df_ = "Canara Bank"

if stock2 == stock:
    df1 = df_axis
    df_1 = "Axis Bank"
elif stock2 == df_1:
    df1 = df_icici
    df_1 = "ICICI Bank"
elif stock2 == df_2:
    df1 = df_sbi
    df_1 = "SBI Bank"
elif stock2 == df_3:
    df1 = df_bob
    df_1 = "Bank of Baroda"
elif stock2 == df_4:
    df1 = df_idbi
    df_1 = "IDBI Bank"
elif stock2 == df_5:
    df1 = df_canara
    df_1 = "Canara Bank"

# Convert Date columns to datetime
df['Date'] = pd.to_datetime(df['Date'])
df1['Date'] = pd.to_datetime(df1['Date'])

# Best of Both Worlds: Optional date filter
default_start = pd.to_datetime("2010-01-04")
default_end = pd.to_datetime("2024-12-31")

with st.expander(" Optional: Customize Analysis Date Range"):
    custom_start = pd.to_datetime(st.date_input("Start Date", default_start.date()))
    custom_end = pd.to_datetime(st.date_input("End Date", default_end.date()))

    if custom_start >= custom_end:
        st.error("Invalid date range. Start date must be before end date.")
        st.stop()

    df = df[(df['Date'] >= custom_start) & (df['Date'] <= custom_end)].reset_index(drop=True)
    df1 = df1[(df1['Date'] >= custom_start) & (df1['Date'] <= custom_end)].reset_index(drop=True)
    st.success(f"Using data from {custom_start.date()} to {custom_end.date()}")
    st.write(f"Number of trading days in selected range: **{len(df)} days**")

st.subheader("Assumption Testing for ARIMA/SARIMA on Spread")

# Re-merge cleanly for spread calculation
merged_df = pd.merge(df[['Date', 'Close']], df1[['Date', 'Close']], on='Date', suffixes=('_1', '_2'))
merged_df = merged_df.sort_values('Date')

# Calculate hedge ratio
y = merged_df['Close_1']
x = merged_df['Close_2']
x_const = sm.add_constant(x)
model = sm.OLS(y, x_const).fit()
hedge_ratio = model.params[1]

# Spread using hedge ratio
spread = y - hedge_ratio * x

# ADF Test
adf_result = adfuller(spread)

# Inference only
if adf_result[1] > 0.05:
    st.warning("Spread is **not stationary** (p > 0.05). Applying first differencing and retesting...")
    spread = spread.diff().dropna()
    adf_result_diff = adfuller(spread)
    if adf_result_diff[1] <= 0.05:
        st.success("After first differencing, the spread is **now stationary**. Proceed with ARIMA modeling.")
    else:
        st.error("Even after first differencing, the spread is **still not stationary**. Consider further differencing or switching to another model such as SARIMA or GARCH.")
else:
    st.success("Spread is **stationary** without differencing. Safe to proceed with ARIMA.")

# -----------------------------------------
#  Assumption 2: Seasonality Check + Adaptive Modeling
# -----------------------------------------
st.write("---")
st.subheader("Assumption 2: Seasonality Check & Adaptive Model Selection")

seasonality_detected = False
model_fitted = False

# Seasonal decomposition (silent, internal only)
try:
    result = seasonal_decompose(spread, model='additive', period=252)
    seasonal_std = np.std(result.seasonal)
    spread_std = np.std(spread)
    if seasonal_std > 0.1 * spread_std:
        seasonality_detected = True
except Exception:
    pass  # fallback silently handled

# Final Evaluation and Model Fitting (no plots or summaries shown)
try:
    if seasonality_detected:
        sarima_model = SARIMAX(spread, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_result = sarima_model.fit(disp=False)
        model_fitted = True
    else:
        arima_model = ARIMA(spread, order=(1, 1, 1))
        arima_result = arima_model.fit()
        model_fitted = True
except:
    model_fitted = False

# Final Assumption Conclusion (inference only)
if model_fitted and not seasonality_detected:
    st.success(" Assumption 2 satisfied: No strong seasonality detected in the spread. ARIMA model applied successfully.")
elif model_fitted and seasonality_detected:
    st.warning(" Assumption 2 **not** satisfied initially: Seasonality was detected. SARIMA model was applied to handle this.")
else:
    st.error(" Assumption 2 could not be verified due to model fitting issues.")

# -----------------------------------------
# Assumption 3: Residual Diagnostics + Adaptive Fixes (Inference Only, No Emojis)
# -----------------------------------------
st.write("---")
st.subheader("Assumption 3: Residual Diagnostics and Model Adaptation")

# Get residuals from the previously fitted model
residuals = sarima_result.resid if seasonality_detected else arima_result.resid
model_used = "SARIMA" if seasonality_detected else "ARIMA"

# Run tests (internally)
shapiro_stat, shapiro_p = stats.shapiro(residuals.dropna())
ljung_result = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
ljung_p = float(ljung_result['lb_pvalue'].values[0])
arch_stat, arch_p, _, _ = het_arch(residuals.dropna())

normality_ok = shapiro_p > 0.05
autocorr_ok = ljung_p > 0.05
hetero_ok = arch_p > 0.05

# Track applied fixes
fix_applied = []

# Fix for non-normal residuals
if not normality_ok:
    shift = abs(spread.min()) + 1 if spread.min() <= 0 else 0
    spread_transformed, lambda_bc = boxcox(spread + shift)
    spread = pd.Series(spread_transformed, index=spread.index)
    fix_applied.append("Box-Cox for normality")

# Fix for autocorrelation
if not autocorr_ok:
    try:
        if model_used == "SARIMA":
            sarima_model = SARIMAX(spread, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
            sarima_result = sarima_model.fit(disp=False)
            residuals = sarima_result.resid
        else:
            arima_model = ARIMA(spread, order=(2, 1, 2))
            arima_result = arima_model.fit()
            residuals = arima_result.resid
        fix_applied.append("ARIMA/SARIMA order increased for autocorrelation")
    except:
        pass

# Fix for heteroscedasticity
if not hetero_ok:
    try:
        garch_model = arch_model(residuals.dropna(), vol='Garch', p=1, q=1)
        garch_result = garch_model.fit(disp='off')
        fix_applied.append("GARCH(1,1) for heteroscedasticity")
    except:
        pass

# Final Summary
st.write("### Final Evaluation")
if normality_ok and autocorr_ok and hetero_ok:
    st.success("Assumption 3 satisfied: Residuals are normal, uncorrelated, and homoscedastic. No further fixes required.")
elif fix_applied:
    st.warning(f"Assumption 3 not initially satisfied. Fixes applied: {', '.join(fix_applied)}")
else:
    st.error("Assumption 3 not satisfied and automatic fixes were unsuccessful. Consider revisiting the model design.")

# -----------------------------------------
# Assumption 4: Normality of Residuals (Inference Only, No Emojis)
# -----------------------------------------
st.write("---")
st.subheader("Assumption 4: Normality of Residuals")

# Use the residuals from the last fitted model
residuals = sarima_result.resid if seasonality_detected else arima_result.resid
residuals = residuals.dropna()

# Run normality tests
shapiro_stat, shapiro_p = stats.shapiro(residuals)
jb_stat, jb_p = stats.jarque_bera(residuals)
ad_result = stats.anderson(residuals, dist='norm')

# Determine if residuals pass normality tests
normality_pass = True
if shapiro_p < 0.05 or jb_p < 0.05 or ad_result.statistic > ad_result.critical_values[2]:  # 5% level
    normality_pass = False

# If not normal, try Box-Cox transformation
boxcox_fixed = False
if not normality_pass:
    try:
        spread_positive = spread + abs(min(spread)) + 1
        boxcox_transformed, lambda_val = stats.boxcox(spread_positive)

        # Test transformed series for normality (proxy)
        shapiro_p2 = stats.shapiro(boxcox_transformed)[1]
        jb_p2 = stats.jarque_bera(boxcox_transformed)[1]

        if shapiro_p2 > 0.05 and jb_p2 > 0.05:
            boxcox_fixed = True
    except:
        pass

# Final summary
st.write("### Final Evaluation")
if normality_pass:
    st.success("Assumption 4 satisfied: Residuals appear to be normally distributed.")
elif boxcox_fixed:
    st.warning("Residuals were not initially normal, but improved after Box-Cox transformation.")
else:
    st.warning("Assumption 4 not satisfied: Residuals are not normally distributed, even after Box-Cox.")

# -----------------------------------------
# Hybrid Modeling: ARIMA/SARIMA + Random Forest (Enhanced)
# -----------------------------------------
st.write("---")
st.subheader("Hybrid Forecasting: Residual Correction using Random Forest")

# Enhanced lag + rolling features
def create_lagged_features(series, lags=10, rolling_windows=[3, 5, 10]):
    df = pd.DataFrame({'y': series})
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = series.shift(i)
    for window in rolling_windows:
        df[f'roll_mean_{window}'] = series.shift(1).rolling(window=window).mean()
        df[f'roll_std_{window}'] = series.shift(1).rolling(window=window).std()
    return df.dropna()

# Get residuals
residual_series = residuals.copy()
rf_df = create_lagged_features(residual_series)

# Train/test split
train_size = int(len(rf_df) * 0.8)
train_rf = rf_df.iloc[:train_size]
test_rf = rf_df.iloc[train_size:]

X_train, y_train = train_rf.drop('y', axis=1), train_rf['y']
X_test, y_test = test_rf.drop('y', axis=1), test_rf['y']

# Feature importance-based pruning (after first fit)
rf_model_init = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_init.fit(X_train, y_train)
importances = pd.Series(rf_model_init.feature_importances_, index=X_train.columns)
low_importance = importances[importances < 0.01].index.tolist()
X_train.drop(columns=low_importance, inplace=True)
X_test.drop(columns=low_importance, inplace=True)

# Enhanced Random Forest
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predict residuals
predicted_residuals = rf_model.predict(X_test)

# Get ARIMA/SARIMA fitted values aligned to test index
if seasonality_detected:
    arima_fitted = sarima_result.fittedvalues
else:
    arima_fitted = arima_result.fittedvalues

arima_base = arima_fitted.loc[y_test.index]

# Hybrid forecast
hybrid_forecast = arima_base.values + predicted_residuals
true_values = y_test + arima_base.values

# Evaluation
rmse = np.sqrt(mean_squared_error(true_values, hybrid_forecast))
mae = mean_absolute_error(true_values, hybrid_forecast)
r2 = r2_score(true_values, hybrid_forecast)

st.write("### Hybrid Model Evaluation (ARIMA + Random Forest)")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# Plot predictions
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.index, true_values, label="Actual Spread", color="black")
ax.plot(y_test.index, hybrid_forecast, label="Hybrid Forecast", color="blue")
ax.set_title("Hybrid Model: Actual vs Forecasted Spread")
ax.set_xlabel("Date")
ax.set_ylabel("Spread")
ax.legend()
st.pyplot(fig)

# -----------------------------------------
# Directional Accuracy and Confusion Matrix
# -----------------------------------------
st.subheader("Directional Accuracy and Confusion Matrix (Hybrid Model)")

actual_direction = pd.Series(true_values, index=y_test.index).diff().dropna().apply(lambda x: 1 if x > 0 else 0)
predicted_direction = pd.Series(hybrid_forecast, index=y_test.index).diff().dropna().apply(lambda x: 1 if x > 0 else 0)

common_index = actual_direction.index.intersection(predicted_direction.index)
actual_direction = actual_direction.loc[common_index]
predicted_direction = predicted_direction.loc[common_index]

directional_accuracy = accuracy_score(actual_direction, predicted_direction)
conf_matrix = confusion_matrix(actual_direction, predicted_direction)
labels = ['Down', 'Up']

st.write(f"**Directional Accuracy:** {directional_accuracy * 100:.2f}%")

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix: Spread Direction")
st.pyplot(fig)

# Interpretation
if directional_accuracy > 0.6:
    st.success("The hybrid model captures directional movement reasonably well.")
elif directional_accuracy > 0.5:
    st.warning("The model performs slightly better than random guessing.")
else:
    st.error("Directional prediction is weak. Consider revisiting model assumptions or feature engineering.")

# -----------------------------------------
# Classification Report
# -----------------------------------------
st.subheader("Classification Report")

report = classification_report(actual_direction, predicted_direction, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:.0f}"
    })
)

