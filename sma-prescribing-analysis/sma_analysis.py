# =============================================================
# SMA Drug Prescribing Bias Analysis
# Author: Jonathan Hill
# 
# METHODOLOGY DEMONSTRATION ONLY
# This script demonstrates the analytical approach used.
# Source data is confidential NHS internal data and is not
# included in this repository.
#
# To run this script you would need access to SMA prescribing
# data with columns: Trust, Intervention (Nusinersen/Risdiplam)
# =============================================================

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
# Source: Internal NHS prescribing data (confidential, not included)
# Expected columns: Trust, Intervention (Nusinersen or Risdiplam)
df = pd.read_excel("sma_prescribing_data.xlsx", sheet_name="Raw data")

# --- 2. Summarise counts by Trust and Intervention ---
summary = df.pivot_table(
    index='Trust',
    columns='Intervention',
    aggfunc='size',
    fill_value=0
)

# --- 3. Totals, proportions, and tariff proxy ---
# Tariff proxy: Trusts prescribing >=60% nusinersen
# are inferred to have a financial tariff incentive
summary['Total'] = summary['Nusinersen'] + summary['Risdiplam']
summary['Risdiplam_Proportion'] = summary['Risdiplam'] / summary['Total']
summary['Tariff_Proxy'] = (
    summary['Nusinersen'] / summary['Total'] >= 0.6
).astype(int)

summary = summary[summary['Total'] > 0].reset_index()

# --- 4. Descriptive Statistics ---
print("Risdiplam Descriptive Stats:")
print("  Mean:    ", summary['Risdiplam'].mean())
print("  Median:  ", summary['Risdiplam'].median())
print("  Std Dev: ", summary['Risdiplam'].std())

print("\nNusinersen Descriptive Stats:")
print("  Mean:    ", summary['Nusinersen'].mean())
print("  Median:  ", summary['Nusinersen'].median())
print("  Std Dev: ", summary['Nusinersen'].std())

print("\nFull summary statistics:")
print(summary[['Risdiplam', 'Nusinersen']].describe())

# --- 5. OLS Regression ---
# Does inferred tariff status predict risdiplam prescribing proportion?
model = smf.ols('Risdiplam_Proportion ~ Tariff_Proxy', data=summary).fit()
print(model.summary())

# --- 6. Visualisation ---
sns.boxplot(
    x='Tariff_Proxy',
    y='Risdiplam_Proportion',
    data=summary,
    palette=['#005EB8', '#AE2573']  # NHS blue and purple
)
plt.xticks([0, 1], ['No Tariff Proxy', 'Likely Tariff Proxy'])
plt.title('Risdiplam Proportion by Inferred Tariff Status')
plt.xlabel('Inferred Tariff for Nusinersen')
plt.ylabel('Risdiplam Proportion')
plt.tight_layout()
plt.savefig("sma_regression_plot.png", dpi=300)
plt.show()
