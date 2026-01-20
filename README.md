# NYC Restaurant Health Inspection Prediction

Predicting restaurant health grades using NYC Department of Health inspection data (296K+ records).

## What This Project Does

I built machine learning models to predict restaurant health inspection outcomes using historical data from NYC's health department. The models predict:

- Restaurant health grades (A, B, or C)
- Whether violations will be critical or non-critical
- Inspection scores based on restaurant characteristics

This helps understand what factors lead to health code violations and poor inspection results.

## The Data

**Source:** NYC Open Data - DOHMH Restaurant Inspection Results

- 296,000+ inspection records
- 27 features including violation codes, cuisine types, locations, and health scores
- Updated daily by NYC Department of Health

**Download the data here:** [NYC Open Data Portal](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)

Save it as `nyc_restaurant_inspections_raw.csv` in the `data/raw/` folder.

## Approach

**1. Data Exploration**
- Looked at grade distributions across boroughs and cuisine types
- Checked for patterns in inspection scores over time
- Analyzed which violations appear most frequently

**2. Data Cleaning**
- Handled missing values and data entry errors
- Dealt with restaurants that haven't been inspected yet (date = 1/1/1900)
- Aggregated multiple violations per inspection

**3. Feature Engineering**
- Processed violation description text
- Created features from inspection dates (day of week, month, year)
- Encoded borough and cuisine information
- Calculated violation frequency per restaurant

**4. Modeling**
- Started with logistic regression as a baseline
- Tried Random Forest and XGBoost
- Tuned hyperparameters and compared performance
- Evaluated using accuracy, precision, recall, and F1-score



## Results

* Currently in progress


## Data Source

Data provided by NYC Department of Health and Mental Hygiene through NYC Open Data.
