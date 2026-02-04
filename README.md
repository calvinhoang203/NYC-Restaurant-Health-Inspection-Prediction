# NYC Restaurant Health Inspection Prediction

Predicting restaurant health grades using machine learning on 296K+ NYC health inspection records.

## What This Does

I built machine learning models to predict restaurant health grades (A, B, or C) using 18 years of inspection data from NYC's Department of Health. The goal was to figure out what actually drives health code compliance - is it location, cuisine type, timing, or just the violations themselves?

## The Data

**Source:** NYC Open Data - DOHMH Restaurant Inspection Results

- 295,831 inspection records spanning 2007-2026
- 30,627 unique restaurants across all 5 boroughs
- Features include violation codes, cuisine types, locations, and inspection scores
- Data updated daily by NYC Department of Health

## My Approach

**Exploration**
- Checked grade distributions: most restaurants (67%) get A grades
- Found that 53% of violations are critical (food safety issues like temperature control, cross-contamination)
- Looked at patterns across boroughs and cuisine types
- Average inspection score is 25 (lower scores are better)

**Cleaning**
- Removed uninspected restaurants (they have placeholder dates of 1/1/1900)
- Kept only graded inspections (A, B, C)
- Combined multiple violation records into single inspections
- Went from 295,831 violation records to 51,839 unique inspections
- Most inspections have 2-3 violations

**Feature Engineering**
- Counted total and critical violations per inspection
- Added time features (month, day of week)
- Encoded cuisine type and borough
- Ended up with 6 features total

**Modeling**
- Started with logistic regression as baseline
- Random Forest performed best
- Main challenge: severe class imbalance (87% of restaurants have A grades)
- Tested on 10,368 inspections

## Results

**Random Forest: 93.8% Accuracy**

The 94% accuracy sounds good but it's misleading because of the class imbalance. If I just predicted "A" for everything, I'd get 87% accuracy without any model.

**Performance by Grade:**

| Grade | Precision | Recall | F1-Score | Count |
|-------|-----------|--------|----------|-------|
| A     | 0.96      | 0.99   | 0.98     | 9,003 |
| B     | 0.72      | 0.64   | 0.68     | 895   |
| C     | 0.75      | 0.59   | 0.66     | 470   |

The model is great at catching A-grade restaurants (99% recall) but struggles with B and C. It misses 36% of B restaurants and 41% of C restaurants, often predicting them as A instead.

**What Actually Matters:**

Feature importance from the Random Forest model:

1. Critical violations - 44%
2. Total violations - 31%
3. Cuisine type - 10%
4. Month - 7%
5. Day of week - 5%
6. Borough - 4%

The two violation features make up 75% of the model. Everything else barely matters. Location, timing, and cuisine type have almost no impact compared to actual food safety violations.

## Main Takeaways

**For restaurant owners:** Reducing critical violations is the only thing that really matters for getting a good grade. Location and timing don't affect your score.

**For the model:** It works well for identifying compliant restaurants but isn't reliable for catching failing ones. The class imbalance makes B and C prediction difficult.

**The real insight:** Food safety violations drive grades. The model basically learned that critical violations predict bad grades, which makes sense but isn't that useful for prediction because we're trying to predict grades before the inspection happens.

## What I Learned

Working with real-world data that's messy and heavily imbalanced. The 94% accuracy looks good on paper but understanding why that number is misleading is more important. 

Class imbalance is a real problem - when 87% of your data is one class, you need to look beyond overall accuracy. Precision and recall for each class tell the real story.

## Limitations

- Only used 6 basic features
- No inspection history (whether restaurants improved or got worse over time)
- Can't tell the difference between B and C grades well
- Missing details about specific violation types
- No info about restaurant characteristics (size, chain vs independent, years in business)

## Next Steps If I Had More Time

- Add features like inspection history and violation types
- Try SMOTE or class weighting to handle the imbalance
- Build separate models for each grade instead of one multi-class model
- Use detailed violation descriptions (currently just counting them)
- Look at temporal patterns for individual restaurants

## Tech Used

Python, pandas, scikit-learn, plotly, streamlit

## Dashboard

Live dashboard: https://calvinhoang203-nyc-restaurant-health-inspection-pred-app-hobo6t.streamlit.app/

## Data Source

NYC Department of Health and Mental Hygiene via NYC Open Data

## License

MIT
