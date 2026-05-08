# Spinal Muscular Atrophy Prescribing Variation Analysis

> ⚠️ **Note:** All figures in this document are illustrative only, created to demonstrate methodology. Actual NHS prescribing data is confidential.

## Project Overview

**Objective:** Demonstrate how statistical analysis can be used to explore variation in SMA treatment patterns across NHS Trusts, using illustrative data and a hypothetical financial tariff variable.

**Clinical Context:**
- **Risdiplam:** Oral therapy, easier to administer, no financial tariff for Trusts
- **Nusinersen:** Intrathecal injection, more complex to administer, some Trusts receive a financial tariff for initiating treatment
- **Question:** Does receiving a financial tariff influence clinical prescribing decisions?

**Tools:** Python, pandas, statsmodels, scipy, seaborn, matplotlib

---

## Illustrative Findings

> These figures are illustrative examples demonstrating the type of analysis conducted, not actual NHS data.

**Illustrative dataset:** 20 NHS Trusts, ~1,200 total approvals over 4 years

### Descriptive Statistics (Illustrative)

| Drug | Mean | Median | Std Dev | Min | Max |
|------|------|--------|---------|-----|-----|
| Drug A (oral, no tariff) | 38.5 | 35.0 | 19.2 | 0 | 78 |
| Drug B (injection, tariff) | 22.1 | 18.0 | 14.8 | 0 | 61 |

**Pattern observed:** High variability between Trusts, with some exclusively prescribing one drug.

---

## Policy Implications

This type of analysis provides evidence for:
- **Reviewing tariff structures** that may incentivize more complex, costly treatments over simpler alternatives
- **Ensuring patient access** is driven by clinical need rather than financial incentives
- **Supporting health equity** by identifying and addressing structural barriers to optimal treatment

---

## Limitations

- **Tariff proxy:** Inferred from prescribing patterns, not confirmed tariff data
- **Confounding variables:** Clinical expertise, patient demographics, local guidelines not controlled for  
- **Small n:** 20 Trusts is a limited sample for regression conclusions
- **Enhancement:** Access to confirmed tariff data and patient-level covariates would strengthen analysis

---

## Skills Demonstrated

- **Statistical testing:** Chi-square test of independence, hypothesis formulation
- **Regression modelling:** OLS, interpretation of coefficients and p-values
- **Model selection:** Justified choice of chi-square over t-test, ANOVA, logistic regression
- **Python:** pandas, scipy, statsmodels, seaborn, matplotlib
- **Healthcare policy analysis:** Understanding clinical and financial incentives in NHS
- **Data governance:** Using aggregate data, suppressing small numbers, maintaining confidentiality

---

## Methodology

### Step 1: Exploratory Data Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load prescribing data
df = pd.read_excel("sma_data.xlsx")

# Summarize by Trust and drug
summary = df.pivot_table(
    index='Trust',
    columns='Intervention',
    aggfunc='size',
    fill_value=0
)

# Descriptive statistics
for drug in summary.columns:
    print(f"\n{drug} stats:")
    print(f"  Mean:    {summary[drug].mean():.2f}")
    print(f"  Median:  {summary[drug].median():.2f}")
    print(f"  Std Dev: {summary[drug].std():.2f}")
    print(f"  Range:   {summary[drug].min()} - {summary[drug].max()}")
```

### Step 2: Hypothesis Testing

**Null Hypothesis (H₀):** No association between NHS Trust and drug prescribed  
**Alternative Hypothesis (H₁):** Financial tariff influences drug selection  
**Significance level:** α = 0.05

```python
import scipy.stats as stats

# Chi-square test of independence
observed = summary.values
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"Degrees of freedom:   {dof}")
print(f"p-value:              {p_value:.2e}")

if p_value < 0.05:
    print("Result: REJECT null hypothesis - significant association found")
```

**Why chi-square?** Both variables (Trust and drug) are categorical. Chi-square tests whether observed distribution deviates from what would be expected if prescribing were random.

**Why not t-test?** A t-test compares means of continuous variables - not appropriate here as we have count data across multiple categories.

### Step 3: Regression Analysis

```python
import statsmodels.formula.api as smf

# Create tariff proxy variable
# Trusts with >60% of one drug likely have the financial tariff
summary['Total'] = summary.sum(axis=1)
summary['Oral_Proportion'] = summary['Drug_A'] / summary['Total']
summary['Tariff_Proxy'] = (
    summary['Drug_B'] / summary['Total'] >= 0.6
).astype(int)

# OLS regression: does tariff status predict oral drug use?
model = smf.ols('Oral_Proportion ~ Tariff_Proxy', data=summary).fit()
print(model.summary())
```

**Model selection reasoning:**
- Outcome (prescription proportion) is **continuous** → OLS regression appropriate
- If outcome were binary (prescribed/not) → logistic regression would be used
- Small sample (20 Trusts) → noted as limitation; larger dataset would strengthen conclusions

### Step 4: Visualization

```python
# Box plot comparing prescribing patterns by tariff status
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
sns.boxplot(
    data=summary,
    x='Tariff_Proxy',
    y='Oral_Proportion',
    ax=axes[0],
    palette=['#005EB8', '#AE2573']  # NHS blue and purple
)
axes[0].set_xticklabels(['No Tariff', 'Likely Tariff'])
axes[0].set_title('Oral Drug Proportion by Inferred Tariff Status')
axes[0].set_ylabel('Proportion of Oral Drug Prescriptions')

# Clustered bar chart by Trust
summary[['Drug_A', 'Drug_B']].plot(
    kind='barh',
    ax=axes[1],
    color=['#005EB8', '#AE2573']
)
axes[1].set_title('Prescriptions by Trust and Drug')
axes[1].set_xlabel('Number of Approvals')

plt.tight_layout()
plt.savefig('sma_prescribing_analysis.png', dpi=300, bbox_inches='tight')
```

---

## Illustrative Results

> These are example findings to demonstrate the type of conclusions this analysis would generate.

**Regression output (illustrative):**
- **Coefficient:** -0.48 (p = 0.002)
- **Interpretation:** Trusts with inferred tariff prescribed approximately **48 percentage points less** of the oral drug
- **R² = 0.41:** Tariff status explains ~41% of prescribing variation

**Statistical test (illustrative):**
- χ² ≈ 142.0, df = 19, p < 0.001
- **Conclusion:** Strong evidence that prescribing patterns differ significantly by Trust, consistent with financial tariff influence

---

