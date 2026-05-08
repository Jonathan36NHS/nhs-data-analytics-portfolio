# Cancer Treatment Demand Forecasting

> ⚠️ **Note:** All figures in this document are illustrative only, 
> created to demonstrate analytical methodology. Actual NHS prescribing 
> data is confidential and not included.

## Project Overview

**Objective:** Forecast monthly prescription volumes for three oncology 
treatments to support NHS stock management and resource planning.

**Clinical Context:**  
Three cancer treatment indications (referred to here as Indication X, Y 
and s) are prescribed across NHS Trusts in England. Senior stakeholders 
needed data-driven confirmation of whether prescribing patterns showed 
seasonality, and a 6-month forward forecast to inform procurement.

**Tools:** Python, pandas, statsmodels, scikit-learn, matplotlib

---

## Illustrative Dataset

> Figures below are entirely fictional, constructed to demonstrate 
> the type of analysis and outputs this project produced.

- **Timeframe:** ~5 years of monthly data
- **Indications:** 3 oncology treatments
- **Records:** Monthly aggregate counts per treatment

---

## Step 1: Descriptive Analytics

Understanding historical patterns before forecasting.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load monthly prescribing data
df = pd.read_excel("oncology_monthly_data.xlsx")

# Aggregate by month and indication
monthly = df.groupby(['month', 'indication'])['count'].sum().unstack()

# Descriptive statistics per indication
for ind in monthly.columns:
    print(f"\n{ind} monthly statistics:")
    print(f"  Mean:    {monthly[ind].mean():.1f}")
    print(f"  Median:  {monthly[ind].median():.1f}")
    print(f"  Mode:    {monthly[ind].mode()[0]:.1f}")
    print(f"  Std Dev: {monthly[ind].std():.1f}")
    print(f"  Min:     {monthly[ind].min():.0f}")
    print(f"  Max:     {monthly[ind].max():.0f}")
```

**Illustrative Results:**

| Indication | Mean | Median | Mode | Std Dev | Min | Max |
|------------|------|--------|------|---------|-----|-----|
| Indication X | 52.3 | 49.0 | 51 | 18.7 | 12 | 94 |
| Indication Y | 31.8 | 30.0 | 28 | 11.2 | 8 | 61 |
| Indication s | 44.6 | 43.0 | 46 | 14.9 | 15 | 78 |

**Interpretation (illustrative):**
- Indication X has the highest average volume with notable variability
- Indication Y is the lowest volume treatment
- Indication s shows the most stable prescribing (tightest distribution relative to mean)

---

## Step 2: Seasonality Testing

Before choosing a forecasting model, test whether seasonality exists.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

for ind in monthly.columns:
    series = monthly[ind].dropna()
    
    # Decompose into trend, seasonal, residual components
    decomposition = seasonal_decompose(
        series, 
        model='additive', 
        period=12  # Monthly data, test for annual seasonality
    )
    
    # Plot components
    fig, axes = plt.subplots(4, 1, figsise=(12, 10))
    decomposition.observed.plot(ax=axes[0], title=f'{ind}: Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.savefig(f'outputs/{ind}_decomposition.png')
```

**Finding (illustrative):** No significant seasonal pattern detected across 
any of the three indications. This ruled out SARIMA and justified simpler 
forecasting models.

---

## Step 3: Model Selection & Training

Three models evaluated on a held-out test set (final 6 months of data).

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import numpy as np
from sklearn.metrics import mean_squared_error

# Train/test split - hold out last 6 months
train = monthly[:-6]
test = monthly[-6:]

results = {}

for ind in monthly.columns:
    train_series = train[ind]
    test_series = test[ind]
    
    # Model 1: Naive (last observed value repeated)
    naive_pred = [train_series.iloc[-1]] * 6
    naive_rmse = np.sqrt(mean_squared_error(test_series, naive_pred))
    
    # Model 2: Simple Exponential Smoothing
    ses = SimpleExpSmoothing(train_series).fit()
    ses_pred = ses.forecast(6)
    ses_rmse = np.sqrt(mean_squared_error(test_series, ses_pred))
    
    # Model 3: Holt's Linear Trend
    holt = Holt(train_series).fit()
    holt_pred = holt.forecast(6)
    holt_rmse = np.sqrt(mean_squared_error(test_series, holt_pred))
    
    results[ind] = {
        'Naive RMSE': naive_rmse,
        'SES RMSE': ses_rmse,
        'Holt RMSE': holt_rmse,
        'Best Model': min(
            [('Naive', naive_rmse), ('SES', ses_rmse), ('Holt', holt_rmse)],
            key=lambda x: x[1]
        )[0]
    }
    
    print(f"\n{ind}:")
    print(f"  Naive RMSE: {naive_rmse:.2f}")
    print(f"  SES RMSE:   {ses_rmse:.2f}")
    print(f"  Holt RMSE:  {holt_rmse:.2f}")
    print(f"  Best model: {results[ind]['Best Model']}")
```

**Why these three models?**

SARIMA was considered but rejected due to no evidence of seasonality. The 
three models chosen represent increasing complexity:

| Model | When Appropriate | Limitation |
|-------|-----------------|------------|
| Naive | Highly stable series, no trend | Ignores all history |
| Simple Exponential Smoothing | Stable series, weights recent data | Cannot capture trend |
| Holt's Linear Trend | Series with gradual upward/downward trend | Cannot capture seasonality |

**Illustrative RMSE Results:**

| Indication | Naive | SES | Holt's | Best Model |
|------------|-------|-----|--------|------------|
| Indication X | 14.2 | 11.8 | 13.1 | **SES** |
| Indication Y | 9.6 | 8.3 | 7.9 | **Holt's** |
| Indication s | 12.4 | 10.7 | 11.2 | **SES** |

---

## Step 4: Final Forecasts

Using the best-performing model per indication.

```python
# Generate final forecasts using best model per indication
forecasts = {}

# Indication X & s: SES
for ind in ['Indication X', 'Indication s']:
    model = SimpleExpSmoothing(monthly[ind]).fit()
    forecasts[ind] = model.forecast(6)

# Indication Y: Holt's (detects subtle downward trend)
holt_y = Holt(monthly['Indication Y']).fit()
forecasts['Indication Y'] = holt_y.forecast(6)

# Compile results
forecast_df = pd.DataFrame(forecasts)
print(forecast_df.round(1))
```

**Illustrative 6-Month Forecast:**

| Month | Indication X | Indication Y | Indication s |
|-------|-------------|-------------|-------------|
| Month +1 | 53.2 | 29.8 | 45.1 |
| Month +2 | 53.2 | 29.1 | 45.1 |
| Month +3 | 53.2 | 28.4 | 45.1 |
| Month +4 | 53.2 | 27.7 | 45.1 |
| Month +5 | 53.2 | 27.0 | 45.1 |
| Month +6 | 53.2 | 26.3 | 45.1 |

**Key Insights (illustrative):**
- **Indication X & s:** Stable - SES forecasts flat continuation
- **Indication Y:** Subtle downward trend detected by Holt's model - flag for procurement review
- **No seasonality confirmed:** Consistent with stakeholder assumption

---

## Step 5: Prescriptive Recommendations

Translating forecasts into actionable NHS planning decisions.

```python
# Flag indications needing attention
for ind in forecast_df.columns:
    trend = forecast_df[ind].iloc[-1] - forecast_df[ind].iloc[0]
    
    if trend < -5:
        print(f"⚠️  {ind}: Downward trend detected - review stock levels")
    elif trend > 5:
        print(f"📈 {ind}: Upward trend - consider increasing procurement")
    else:
        print(f"✅ {ind}: Stable - maintain current stock levels")
```

**Recommendations (illustrative):**
- ✅ **Indication X:** Maintain current procurement (~53 units/month)
- ⚠️ **Indication Y:** Gradual decline - reduce stock over next 6 months to avoid waste
- ✅ **Indication s:** Stable - no procurement changes needed

---

## Impact

**Business value delivered:**
- Confirmed no seasonality - simplified procurement scheduling
- Identified declining trend in one indication - prevented potential overstock
- Provided 6-month evidence base for supplier negotiations
- Reduced reliance on subjective stakeholder assumptions

---

## Skills Demonstrated

- **Time series forecasting:** Naive, SES, Holt's Linear Trend
- **Model selection:** RMSE-based evaluation on held-out test set
- **Seasonality testing:** Decomposition analysis (statsmodels)
- **Python:** pandas, statsmodels, scikit-learn, matplotlib
- **Prescriptive analytics:** Translating model outputs into recommendations
- **Healthcare planning:** Drug procurement, stock management, NHS resource allocation
- **Statistical communication:** Presenting complex findings to non-technical stakeholders

## Limitations & Future Enhancements

- Tariff proxy is inferred, not confirmed
- Small number of indications limits generalisability
- Enhancement: SARIMA if seasonality emerges in future data
- Enhancement: Prophet model for automatic trend/seasonality detection
- Enhancement: Monte Carlo simulation for forecast confidence intervals

---

*Methodology based on Level 4 Data Analyst Apprenticeship project, NHS England 2024-2025. All figures are illustrative.*
