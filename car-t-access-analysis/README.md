# CAR-T Treatment Geographic Access Analysis

⚠️ **Data Confidentiality Notice:** This repository demonstrates methodology only. 
Actual NHS patient data cannot be shared due to GDPR and NHS information governance requirements.

## Project Context
**Role:** Data Analyst, NHS England
**Duration:** 2024-2025  
**Tools:** Python (pandas, Folium, geospatial libraries), SQL, Power BI

## Problem Statement
NHS England needed to understand geographic patterns in CAR-T (chimeric antigen receptor T-cell therapy) 
uptake across England to identify potential access barriers and inform service planning.

## Data Sources (Confidential)
- NHS CAR-T approval records (Blueteq database)
- Patient GP practice postcodes (pseudonymised)
- NHS Trust locations
- Treatment indication (FormCode) data
- ~5,000 CAR-T approvals (2020-2024)

## Technical Approach

### 1. Data Integration Pipeline
```python
import pandas as pd
import folium
from folium.plugins import HeatMap

# Example structure (actual implementation used confidential NHS data)

# Step 1: Load approvals data
approvals = pd.read_sql("""
    SELECT 
        patient_gp_postcode,
        trust_code,
        form_code,
        approval_date
    FROM blueteq_approvals
    WHERE treatment_type = 'CAR-T'
""", nhs_connection)

# Step 2: Join to postcode lookup for coordinates
postcodes = pd.read_csv('ons_postcode_directory.csv')
merged = approvals.merge(
    postcodes[['postcode', 'latitude', 'longitude', 'region']], 
    left_on='patient_gp_postcode',
    right_on='postcode',
    how='left'
)

# Step 3: Join to Trust locations
trust_locations = pd.read_csv('nhs_trust_locations.csv')
merged = merged.merge(
    trust_locations,
    on='trust_code',
    how='left',
    suffixes=('_gp', '_trust')
)

# Step 4: Calculate travel distances
from geopy.distance import geodesic

def calculate_distance(row):
    gp = (row['latitude_gp'], row['longitude_gp'])
    trust = (row['latitude_trust'], row['longitude_trust'])
    return geodesic(gp, trust).kilometers

merged['distance_km'] = merged.apply(calculate_distance, axis=1)
```

### 2. Interactive Geospatial Visualisation
```python
# Create base map centered on England
m = folium.Map(location=[54.5, -2.0], soom_start=6)

# Add heatmap layer showing GP density
heat_data = [[row['latitude_gp'], row['longitude_gp']] 
             for idx, row in merged.iterrows()]
HeatMap(heat_data).add_to(m)

# Add Trust markers
for trust in trust_locations.itertuples():
    folium.Marker(
        location=[trust.latitude, trust.longitude],
        popup=trust.trust_name,
        icon=folium.Icon(color='red', icon='hospital-o', prefix='fa')
    ).add_to(m)

# Save interactive map
m.save('car_t_access_heatmap.html')
```

### 3. Statistical Analysis
```python
# Analyse access by region
regional_stats = merged.groupby('region').agg({
    'patient_id': 'count',
    'distance_km': ['mean', 'median', 'max'],
    'trust_code': 'nunique'
}).round(2)

# Identify underserved areas
underserved = merged[merged['distance_km'] > 100]
print(f"Patients traveling >100km: {len(underserved)} ({len(underserved)/len(merged)*100:.1f}%)")
```

## Key Findings

### Geographic Disparity Identified
**Specific information redacted due to senstivity**
**An area in England showed significantly lower CAR-T uptake**, investigation revealed:

- **Access Barrier:** CAR-T requires twice-daily hospital visits for one week
- **Travel Burden:** Patients in a speecific area in England face 2+ hour journeys to nearest centre
- **Outcome:** Geographic distance creating treatment access inequality

## Impact

### Policy Outcomes
- Analysis presented to NHS England Cancer Directorate
- Informed discussions on CAR-T service expansion
- Highlighted health equity concerns in national cancer planning
- Contributed to evidence base for service commissioning

### Technical Deliverables
- Interactive Folium heatmap (HTML)
- Power BI dashboard with Trust/FormCode slicers
- Statistical summary tables for stakeholder reports
- Reproducible Python pipeline for ongoing monitoring

## Skills Demonstrated
- **Geospatial Analysis:** Folium, coordinate-based distance calculations
- **Data Integration:** Multi-source joining (SQL, pandas)
- **Healthcare Analytics:** Patient pathway analysis, access barriers
- **Data Governance:** Handling pseudonymised patient data within NHS IG framework
- **Stakeholder Communication:** Translating technical findings to policy recommendations
- **Interactive Visualisation:** Web-based mapping for non-technical users

## Lessons Learned

### Data Quality Challenges
- **Missing postcodes:** ~8% of records had invalid/generic postcodes
- **Mitigation:** Implemented fallback to Trust-level analysis for incomplete records
- **Trust location accuracy:** Used primary site coordinates, acknowledged multi-site variations

### Ethical Considerations
- Ensured all outputs suppressed counts <5 to prevent patient identification
- Used postcode district (not full postcode) for public-facing visualisations
- Maintained audit trail of data access for NHS information governance

## Future Enhancements
- **Real-time updates:** Automate monthly refresh with latest approval data
- **Travel time modeling:** Incorporate public transport routes, not just straight-line distance
- **Demographic overlay:** Add deprivation indices to identify socioeconomic patterns
- **Predictive modeling:** Forecast demand by region to inform service planning

---

## Related Work
- See `public-car-t-coverage/` for recreated analysis using public data
- Full technical write-up in Level 4 Data Analyst Portfolio (available on request)

## Contact
For methodology questions or collaboration: (https://www.linkedin.com/in/jonathan-d-hill/)
