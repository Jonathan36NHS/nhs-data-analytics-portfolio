# Breast Cancer Treatment Demand Forecasting

Time series forecasting of monthly prescriptions for three breast cancer indications (ABEM2, PAL2, RIB2) to support NHS stock management and resource planning.

⚠️ **Data Confidentiality Notice:** Methodology demonstration only. Actual NHS prescribing data is confidential. The following data is synthetic and does not reflect real patient or prescribing records.

## Objective
Forecast next 6 months of demand for ABEM2, PAL2, and RIB2 to optimise NHS drug procurement and reduce stockouts/waste.

## Data
- **Timeframe:** September 2019 - May 2025 (69 months)
- **Indications:** 3 breast cancer treatments
- **Total approvals:** 3, 206 prescriptions

## Methodology

### 1. Descriptive Analytics
```python
# Monthly aggregation
monthly_counts = df.groupby(['Month', 'FormCode']).size().unstack(fill_value=0)

# Summary statistics per drug
for drug in ['ABEM2', 'PAL2', 'RIB2']:
    print(f"\n{drug} statistics:")
    print(f"Mean: {monthly_counts[drug].mean():.2f}")
    print(f"Median: {monthly_counts[drug].median():.2f}")
    print(f"Mode: {monthly_counts[drug].mode().values[0]}")
    print(f"Std Dev: {monthly_counts[drug].std():.2f}")
```

**Results:**
- **ABEM2:** Mean=51.43, Median=47, Mode=52, SD=31.63
- **PAL2:** Mean=74.61, Median=71, Mode=56, SD=36.13 (highest variability)
- **RIB2:** Mean=38.00, Median=38, Mode=49, SD=25.85

### 2. Model Selection & Training

**Models Evaluated:**
1. **Naive Forecast:** Next month = last observed month
2. **Simple Exponential Smoothing:** Weighted average favoring recent values
3. **Holt's Linear Trend:** Accounts for gradual increases/decreases

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt

# Train-test split (last 6 months held out for validation)
train = monthly_counts[:-6]
test = monthly_counts[-6:]

# Fit models for each drug
for drug in ['ABEM2', 'PAL2', 'RIB2']:
    # Naive
    naive_forecast = [train[drug].iloc[-1]] * 6
    
    # Exponential Smoothing
    es_model = SimpleExpSmoothing(train[drug]).fit()
    es_forecast = es_model.forecast(6)
    
    # Holt's Linear Trend
    holt_model = Holt(train[drug]).fit()
    holt_forecast = holt_model.forecast(6)
```

### 3. Model Evaluation (RMSE)

```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE for each model
for drug in ['ABEM2', 'PAL2', 'RIB2']:
    naive_rmse = np.sqrt(mean_squared_error(test[drug], naive_forecast))
    es_rmse = np.sqrt(mean_squared_error(test[drug], es_forecast))
    holt_rmse = np.sqrt(mean_squared_error(test[drug], holt_forecast))
    
    print(f"{drug}: Naive RMSE={naive_rmse:.2f}, "
          f"Simple Exp Smoothing RMSE={es_rmse:.2f}, "
          f"Holt's Linear RMSE={holt_rmse:.2f}")
```

**Results:**
- **ABEM2:** Best model = **Simple Exponential Smoothing** (RMSE=21.55)
- **PAL2:** Best model = **Simple Exponential Smoothing** (RMSE=8.64)
- **RIB2:** Best model = **Holt's Linear Trend** (RMSE=5.78)

### 4. Final Forecasts (June-November 2025)

| Month | ABEM2 | PAL2 | RIB2 |
|-------|-------|------|------|
| Jun 2025 | 36.7 | 55.0 | 50.8 |
| Jul 2025 | 36.7 | 55.0 | 50.4 |
| Aug 2025 | 36.7 | 55.0 | 50.0 |
| Sep 2025 | 36.7 | 55.0 | 49.6 |
| Oct 2025 | 36.7 | 55.0 | 49.2 |
| Nov 2025 | 36.7 | 55.0 | 48.8 |

**Key Insights:**
- **No seasonality detected** - stable monthly patterns
- **RIB2 shows slight downward trend** (Holt's captures this)
- **PAL2 remains highest volume** with moderate variability

## Business Impact

**Stock Management:**
- Informed procurement: ~140 units/month total demand
- Reduced waste: No need for large safety stocks given stability
- Improved planning: 6-month visibility for budgeting

**Stakeholder Value:**
- Confirmed stakeholder assumption of no seasonality (data-driven validation)
- Enabled consistent stock levels throughout 2025
- Provided confidence intervals for risk management

## Skills Demonstrated
- **Time Series Analysis:** ARIMA, exponential smoothing, Holt's method
- **Model Selection:** RMSE-based comparison, choosing appropriate models
- **Python:** statsmodels, pandas, sklearn
- **Forecasting:** Multi-step ahead prediction
- **Healthcare Analytics:** Drug demand planning

## Lessons Learned
- **SARIMA not needed:** Lack of seasonality meant simpler models performed best
- **Drug-specific patterns:** Each treatment required individual model selection
- **Validation critical:** Hold-out testing prevented overfitting

---
Code: `code/breast_cancer_forecasting.py`
