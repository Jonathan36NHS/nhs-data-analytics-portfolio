# =============================================================
# CAR-T Patient Journey Geographic Access Analysis
# Author: Jonathan Hill
#
# METHODOLOGY DEMONSTRATION ONLY
# This script demonstrates the analytical approach used in an
# NHS England CAR-T access analysis project.
#
# Source data is confidential NHS internal data and is NOT
# included in this repository. To run this script you would
# need:
#   - CAR-T approval records with GP practice postcodes
#   - ONS Open Postcode Geo dataset (publicly available)
#   - NHS Trust postcode data (publicly available via ODS)
#
# ONS Postcode data available at:
# https://geoportal.statistics.gov.uk
# =============================================================

import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# --- 1. Load Data ---
# Source: Internal NHS CAR-T approvals (confidential, not included)
# Expected columns: PatientPracticePostCode, Trust, FormIdentifierCode, FormName
car_t = pd.read_csv('data/cart_approvals.csv')

# Source: ONS Open Postcode Geo (publicly available)
postcode = pd.read_csv('data/open_postcode_geo.csv', header=None)

print(f"CAR-T approvals: {car_t.shape}")
print(f"Postcode records: {postcode.shape}")

# --- 2. Assign Column Names to Postcode Dataset ---
postcode.columns = [
    'postcode', 'status', 'size', 'easting', 'northing', 'index', 'country',
    'latitude', 'longitude', 'pcd_nospace', 'pcd1', 'pcd2', 'pcds', 'pcdt',
    'pcd_area', 'pcd_district', 'pcd_sector'
]

# --- 3. Standardise Postcodes for Merging ---
car_t['PatientPracticePostCode'] = (
    car_t['PatientPracticePostCode'].str.replace(' ', '').str.upper()
)
postcode['postcode'] = postcode['postcode'].str.replace(' ', '').str.upper()

# --- 4. Merge GP Practice Postcodes to Coordinates ---
merged = car_t.merge(
    postcode,
    left_on='PatientPracticePostCode',
    right_on='postcode',
    how='left'
)

missing_lat = merged['latitude'].isna().sum()
print(f"{missing_lat} rows have missing latitude (unmatched postcodes)")

# --- 5. Handle Generic Codes (NI, Scotland, Wales) ---
# Patients from outside England receive representative coordinates
generic_coords = {
    'NIGP':               {'latitude': 54.5973, 'longitude': -5.9301},  # Belfast
    'SCOTLANDGPPRACTICE': {'latitude': 55.9533, 'longitude': -3.1883},  # Edinburgh
    'WALESGPPRACTICE':    {'latitude': 51.4816, 'longitude': -3.1791},  # Cardiff
    'WALESPRACTICE':      {'latitude': 51.4816, 'longitude': -3.1791},  # Cardiff variant
}
for code, coords in generic_coords.items():
    mask = merged['PatientPracticePostCode'] == code
    merged.loc[mask, 'latitude'] = coords['latitude']
    merged.loc[mask, 'longitude'] = coords['longitude']

# --- 6. Load and Filter NHS Trust Locations ---
# Source: NHS Organisation Data Service (publicly available)
trusts_raw = pd.read_csv('data/nhse_trust_postcodes.csv', header=None)
trusts_raw.columns = [
    'OrganisationCode', 'TrustName', 'Unknown1', 'Unknown2', 'Unknown3',
    'Hospital', 'Street', 'City', 'NA', 'Postcode', 'OpenDate', 'Status'
]
trusts_for_mapping = trusts_raw[['TrustName', 'Postcode']].copy()

# --- 7. Filter to CAR-T Centres ---
cart_centres = [
    "TRUST A",
    "TRUST B",
    "TRUST C",
    "TRUST D",
    "TRUST E",
    "TRUST F",
    "TRUST G",
    "TRUST H",
    TRUST I",
    "TRUST J",
    "TRUST K",
    "TRUST L",
    "TRUST M",
    "TRUST N",
    "TRUST O",
    "TRUST P",
    "TRUST Q",
    "TRUST R",
    "TRUST S",
    "TRUST T",
    "TRUST U"
]

trusts_for_mapping['TrustName_clean'] = (
    trusts_for_mapping['TrustName'].str.upper().str.strip()
)
cart_trusts = trusts_for_mapping[
    trusts_for_mapping['TrustName_clean'].isin(cart_centres)
].copy()
cart_trusts['Postcode'] = cart_trusts['Postcode'].str.replace(' ', '').str.upper()

# --- 8. Get Coordinates for CAR-T Centres ---
cart_trusts_coords = cart_trusts.merge(
    postcode,
    left_on='Postcode',
    right_on='postcode',
    how='left'
)
print(f"CAR-T centres matched: {cart_trusts_coords.shape[0]}")

# --- 9. Build Full Merged Dataset ---
car_t['Trust'] = car_t['Trust'].replace({
    'Manchester University FT – Oxford Road': 'MANCHESTER UNIVERSITY NHS FOUNDATION TRUST',
    'OXFORD UNIVERSITY HOSPITALS NHS TRUST': 'OXFORD UNIVERSITY HOSPITALS NHS FOUNDATION TRUST'
})

full = car_t.merge(
    cart_trusts_coords[['TrustName', 'latitude', 'longitude']],
    left_on='Trust',
    right_on='TrustName',
    how='left',
    suffixes=('', '_Trust')
)

full = full.merge(
    postcode[['postcode', 'latitude', 'longitude']],
    left_on='PatientPracticePostCode',
    right_on='postcode',
    how='left',
    suffixes=('', '_Practice')
)

# Re-apply generic coordinates after second merge
for code, coords in generic_coords.items():
    mask = full['PatientPracticePostCode'] == code
    full.loc[mask, 'latitude_Practice'] = coords['latitude']
    full.loc[mask, 'longitude_Practice'] = coords['longitude']

# --- 10. Filter to Valid Rows ---
valid = full[
    full['PatientPracticePostCode'].notna() &
    (full['PatientPracticePostCode'] != 'UNKNOWN')
].copy()
valid = valid.dropna(subset=[
    'latitude', 'longitude', 'latitude_Practice', 'longitude_Practice'
])

for col in ['latitude', 'longitude', 'latitude_Practice', 'longitude_Practice']:
    valid[col] = pd.to_numeric(valid[col], errors='coerce')
valid = valid.dropna(subset=[
    'latitude', 'longitude', 'latitude_Practice', 'longitude_Practice'
])

print(f"Rows ready for mapping: {valid.shape[0]}")

# --- 11. Calculate Haversine Distances ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

valid['distance_km'] = haversine(
    valid['latitude_Practice'], valid['longitude_Practice'],
    valid['latitude'], valid['longitude']
)

# --- 12. Summary Statistics by Trust ---
trust_commute_summary = (
    valid.groupby('TrustName')['distance_km']
    .agg(['mean', 'median', 'max', 'count'])
    .sort_values('mean', ascending=False)
    .reset_index()
)
print("\nCommute distance by Trust:")
print(trust_commute_summary)

max_commute_row = valid.loc[valid['distance_km'].idxmax()]
print(f"\nFurthest journey: {max_commute_row['distance_km']:.1f} km")
print(f"To: {max_commute_row['TrustName']}")

# --- 13. Interactive Map - By Trust and FormCode ---
formcode_col = 'FormIdentifierCode'
trust_col = 'TrustName'

m = folium.Map(location=[54.5, -4], zoom_start=6)
main_fg = folium.FeatureGroup(name="All Journeys", show=True)
m.add_child(main_fg)

ordered_trusts = sorted(valid[trust_col].unique())
ordered_formcodes = sorted(valid[formcode_col].dropna().unique())

# Create layer groups
trust_layers = {}
for trust in ordered_trusts:
    fg = folium.FeatureGroup(name=f"Trust: {trust}", show=False)
    m.add_child(fg)
    trust_layers[trust] = fg

formcode_layers = {}
for fc in ordered_formcodes:
    fg = folium.FeatureGroup(name=f"FormCode: {fc}", show=False)
    m.add_child(fg)
    formcode_layers[fc] = fg

# Add CAR-T centre markers
all_trust_centre_rows = valid.drop_duplicates([trust_col, 'latitude', 'longitude'])
for _, row in all_trust_centre_rows.sort_values(trust_col).iterrows():
    marker = folium.Marker(
        [row['latitude'], row['longitude']],
        popup=f"Centre: {row[trust_col]}",
        icon=folium.Icon(color='red', icon='hospital', prefix='fa')
    )
    main_fg.add_child(marker)
    trust_layers[row[trust_col]].add_child(
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"Centre: {row[trust_col]}",
            icon=folium.Icon(color='red', icon='hospital', prefix='fa')
        )
    )
    for fc in ordered_formcodes:
        formcode_layers[fc].add_child(
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"Centre: {row[trust_col]}",
                icon=folium.Icon(color='red', icon='hospital', prefix='fa')
            )
        )

# Add trust summary popups
trust_summary = (
    valid.groupby(trust_col)['distance_km']
    .agg(['mean', 'max', 'count'])
    .reset_index()
)
for _, row in trust_summary.iterrows():
    trust = row[trust_col]
    trow = valid.loc[valid[trust_col] == trust].iloc[0]
    trust_layers[trust].add_child(
        folium.Marker(
            [trow['latitude'], trow['longitude']],
            popup=(
                f"<b>{trust}</b><br>"
                f"Approvals: {int(row['count'])}<br>"
                f"Avg Commute: {row['mean']:.1f} km<br>"
                f"Longest: {row['max']:.1f} km"
            ),
            icon=folium.Icon(color='orange', icon='info-sign')
        )
    )

# Add formcode summary popups
formcode_summary = (
    valid.groupby(formcode_col)['distance_km']
    .agg(['mean', 'max', 'count'])
    .reset_index()
)
for _, row in formcode_summary.iterrows():
    fc = row[formcode_col]
    formcode_layers[fc].add_child(
        folium.Marker(
            [54.5, -4],
            popup=(
                f"<b>FormCode: {fc}</b><br>"
                f"Approvals: {int(row['count'])}<br>"
                f"Avg Commute: {row['mean']:.1f} km<br>"
                f"Longest: {row['max']:.1f} km"
            ),
            icon=folium.Icon(color='green', icon='medkit', prefix='fa')
        )
    )

folium.LayerControl(collapsed=False).add_to(m)

# Add patient journeys (GP circles + PolyLines to Trust)
for _, row in valid.iterrows():
    trust = row[trust_col]
    fc = row[formcode_col]
    popup_text = (
        f"GP Postcode District: {row['PatientPracticePostCode'][:4]}<br>"
        f"Trust: {trust}<br>"
        f"FormCode: {fc}<br>"
        f"Distance: {row['distance_km']:.1f} km"
    )
    for fg in [main_fg, trust_layers[trust], formcode_layers.get(fc, main_fg)]:
        folium.CircleMarker(
            [row['latitude_Practice'], row['longitude_Practice']],
            radius=3, color='blue', fill=True, fill_opacity=0.6,
            popup=popup_text
        ).add_to(fg)
        folium.PolyLine(
            locations=[
                [row['latitude_Practice'], row['longitude_Practice']],
                [row['latitude'], row['longitude']]
            ],
            color='green', weight=1, opacity=0.3
        ).add_to(fg)

# Save map
m.save('outputs/cart_patient_journeys.html')
print("Map saved to outputs/cart_patient_journeys.html")

# --- 14. Entity Relationship Diagram ---
entities = {
    'A': {
        'title': 'CAR-T Approvals',
        'fields': [
            'ApprovalID (PK)', 'PatientPracticePostCode',
            'TrustName', 'FormCode', 'Date', 'Intervention', 'Month', 'Year'
        ],
        'pk': ['ApprovalID']
    },
    'B': {
        'title': 'Postcode Directory',
        'fields': [
            'Postcode (PK)', 'Latitude', 'Longitude', 'Region', 'Country'
        ],
        'pk': ['Postcode']
    },
    'C': {
        'title': 'NHS Trust Locations',
        'fields': [
            'TrustName (PK)', 'Postcode', 'Latitude',
            'Longitude', 'OrganisationCode'
        ],
        'pk': ['TrustName']
    },
    'D': {
        'title': 'Merged Dataset',
        'fields': [
            'ApprovalID', 'PatientPracticePostCode', 'TrustName',
            'FormCode', 'Date', 'Latitude_GP', 'Longitude_GP',
            'Latitude_Trust', 'Longitude_Trust', 'TravelDistance_km (derived)'
        ],
        'pk': []
    }
}

positions = {'A': (0, 4), 'B': (0, 0), 'C': (0, -4), 'D': (7, 0)}
edges = [('A', 'D', 'PatientPracticePostCode'), ('B', 'D', 'Postcode'), ('C', 'D', 'TrustName')]

G = nx.DiGraph()
for k in entities:
    G.add_node(k)
for src, dst, lbl in edges:
    G.add_edge(src, dst, label=lbl)

fig, ax = plt.subplots(figsize=(15, 10))
ax.axis('off')

for k in entities:
    x, y = positions[k]
    num_fields = len(entities[k]['fields'])
    box_width, title_height, row_height = 3.4, 0.7, 0.32
    box_height = title_height + num_fields * row_height + 0.18

    box = FancyBboxPatch(
        (x - box_width/2, y - box_height/2), box_width, box_height,
        boxstyle="round,pad=0.07",
        edgecolor="navy", facecolor="lightblue", linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y + box_height/2 - title_height/2, entities[k]['title'],
            ha="center", va="center", fontsize=15, fontweight='bold')

    field_y = y + box_height/2 - title_height - 0.06
    for idx, field in enumerate(entities[k]['fields']):
        is_pk = field.split(' ')[0] in entities[k]['pk']
        ax.text(x, field_y - idx * row_height, field,
                ha="center", va="center", fontsize=11,
                fontweight='bold' if is_pk else 'normal',
                color='darkred' if is_pk else 'black')

for src, dst, lbl in edges:
    src_pos, dst_pos = positions[src], positions[dst]
    x_delta = dst_pos[0] - src_pos[0]
    y_delta = dst_pos[1] - src_pos[1]
    norm = (x_delta**2 + y_delta**2)**0.5
    offset_src = (src_pos[0] + x_delta/norm * 1.7, src_pos[1] + y_delta/norm * 0.7)
    offset_dst = (dst_pos[0] - x_delta/norm * 1.7, dst_pos[1] - y_delta/norm * 0.7)
    ax.annotate("", xy=offset_dst, xytext=offset_src,
                arrowprops=dict(arrowstyle="->", color="brown", lw=2,
                                shrinkA=12, shrinkB=12))
    mid_x = (offset_src[0] + offset_dst[0]) / 2
    mid_y = (offset_src[1] + offset_dst[1]) / 2
    ax.text(mid_x, mid_y, lbl, fontsize=11, color='brown',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.xlim(-3, 10)
plt.ylim(-6, 6)
plt.tight_layout()
plt.savefig('outputs/erd_diagram.png', dpi=300, bbox_inches='tight')
plt.show()
