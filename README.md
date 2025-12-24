# Pair-Trading-Analysis
An interactive Streamlit dashboard for quantitative Pairs Trading. Features automated cointegration testing, adaptive SARIMA modeling, and a Hybrid Random Forest residual-correction engine for financial time-series forecasting.
## Key Features

* **Dynamic Pair Selection**: Compare stocks across major institutions including Axis Bank, ICICI Bank, SBI, Bank of Baroda, IDBI, and Canara Bank.
* **Statistical Arbitrage Engine**: Automates the calculation of hedge ratios using OLS regression and generates the trading spread.
* **Rigorous Hypothesis Testing**:
    * **Stationarity**: Augmented Dickey-Fuller (ADF) test.
    * **Normality**: Shapiro-Wilk and Jarque-Bera tests.
    * **Autocorrelation**: Ljung-Box test.
    * **Heteroscedasticity**: ARCH test.
* **Adaptive Modeling Pipeline**: Automatically detects seasonality to switch between ARIMA and SARIMA models.
* **Hybrid ML Correction**: Utilizes a Random Forest Regressor to predict and correct residuals, capturing non-linear patterns in the spread.
* **Interactive Dashboard**: Built with Streamlit to allow for custom date ranges and real-time visualization of results.



## Technical Methodology

1. **Spread Calculation**: $Spread = Stock A - (Hedge Ratio \times Stock B)$.
2. **Assumption Validation**: The engine performs a four-stage diagnostic check on residuals to ensure model validity.
3. **Refinement**: Automatically applies Box-Cox transformations or GARCH(1,1) modeling if residuals fail standard assumptions.
4. **Evaluation**: Provides a full classification report and confusion matrix to measure the model's directional accuracy.



## Tech Stack

* **Language**: Python
* **Framework**: Streamlit
* **Statistics/ML**: Statsmodels, Scikit-Learn, Arch, SciPy
* **Visualization**: Matplotlib, Seaborn
