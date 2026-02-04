import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NYC Restaurant Inspection Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .big-number {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 18px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    import os
    data_path = 'data/inspections_clean.csv'
    
    # Debug: List current directory contents
    current_dir = os.listdir('.')
    data_dir_exists = os.path.exists('data')
    data_dir_contents = []
    if data_dir_exists:
        data_dir_contents = os.listdir('data')
    
    # Check if file exists first
    if not os.path.exists(data_path):
        # Return debug info in error message
        st.error(f"""
        **Debug Info:**
        - Current directory: {os.getcwd()}
        - Files in current dir: {', '.join(current_dir[:10])}
        - Data directory exists: {data_dir_exists}
        - Data directory contents: {', '.join(data_dir_contents) if data_dir_contents else 'None'}
        - Looking for: {data_path}
        """)
        return None
    
    try:
        df = pd.read_csv(data_path)
        df['INSPECTION_DATE'] = pd.to_datetime(df['INSPECTION_DATE'])
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load encoders
        with open('models/cuisine_encoder.pkl', 'rb') as f:
            cuisine_encoder = pickle.load(f)
        
        with open('models/boro_encoder.pkl', 'rb') as f:
            boro_encoder = pickle.load(f)
        
        return model, cuisine_encoder, boro_encoder
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.warning(f"Model file not found or error loading model: {str(e)}")
        return None, None, None

df = load_data()
model, cuisine_encoder, boro_encoder = load_model()

# Check if data is loaded
if df is None:
    st.error("""
    ## ⚠️ Data File Not Found
    
    The required data file `data/inspections_clean.csv` is not found in the repository.
    
    **To fix this:**
    
    1. **If you have the data file locally:**
       - Ensure the `data/` directory exists in your repository
       - Add `inspections_clean.csv` to the `data/` directory
       - Commit and push the file to your repository
    
    2. **If you need to generate the data:**
       - Run the `02_clean.ipynb` notebook to process the raw data
       - This will create `data/inspections_clean.csv`
       - Then commit and push the file
    
    3. **For Streamlit Cloud deployment:**
       - The data file needs to be in your GitHub repository
       - Check that `data/inspections_clean.csv` is not in `.gitignore`
       - If it's too large, consider using Streamlit's file uploader or external storage
    
    **Note:** The `/data` directory is currently in `.gitignore`. You may need to:
    - Remove `/data` from `.gitignore` if you want to commit the data file
    - Or use a different approach like storing data in a cloud storage service
    """)
    st.stop()

# Sidebar filters
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

# Borough filter
boroughs = ['All'] + sorted(df['BORO'].dropna().unique().tolist())
selected_borough = st.sidebar.selectbox("Borough", boroughs)

# Cuisine filter
cuisines = ['All'] + sorted(df['CUISINE'].dropna().unique().tolist())
selected_cuisine = st.sidebar.selectbox("Cuisine Type", cuisines)

# Date range
min_date = df['INSPECTION_DATE'].min().date()
max_date = df['INSPECTION_DATE'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
filtered_df = df.copy()
if selected_borough != 'All':
    filtered_df = filtered_df[filtered_df['BORO'] == selected_borough]
if selected_cuisine != 'All':
    filtered_df = filtered_df[filtered_df['CUISINE'] == selected_cuisine]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['INSPECTION_DATE'].dt.date >= date_range[0]) & 
        (filtered_df['INSPECTION_DATE'].dt.date <= date_range[1])
    ]

st.sidebar.markdown("---")
st.sidebar.info(f"**{len(filtered_df):,}** inspections matched")

# Main dashboard
st.title("🍽️ NYC Restaurant Inspection Analytics")
st.markdown("### Machine Learning Analysis of Restaurant Health Grades")
st.markdown("---")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Inspections",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df)/len(df)*100:.1f}% of data"
    )

with col2:
    avg_score = filtered_df['SCORE'].mean()
    st.metric(
        label="Average Score",
        value=f"{avg_score:.1f}",
        delta=f"{avg_score - df['SCORE'].mean():.1f} vs overall",
        delta_color="inverse"
    )

with col3:
    a_grade_pct = (filtered_df['GRADE'] == 'A').sum() / len(filtered_df) * 100
    st.metric(
        label="A Grade Rate",
        value=f"{a_grade_pct:.1f}%",
        delta=f"{a_grade_pct - 86.8:.1f}% vs overall"
    )

with col4:
    critical_pct = filtered_df['CRITICAL_VIOLATIONS'].sum() / filtered_df['TOTAL_VIOLATIONS'].sum() * 100
    st.metric(
        label="Critical Violation Rate",
        value=f"{critical_pct:.1f}%",
        delta=None
    )

st.markdown("---")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Explore Data", "🤖 Model Performance", "🎯 Make Prediction"])

with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Grade distribution pie
        grade_counts = filtered_df['GRADE'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=grade_counts.index,
            values=grade_counts.values,
            marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c']),
            hole=0.4
        )])
        fig_pie.update_layout(
            title="Grade Distribution",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Score distribution histogram
        fig_hist = px.histogram(
            filtered_df,
            x='SCORE',
            nbins=40,
            title="Score Distribution",
            color_discrete_sequence=['#3498db']
        )
        fig_hist.add_vline(
            x=filtered_df['SCORE'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {filtered_df['SCORE'].mean():.1f}"
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Grades by borough
        boro_grade = pd.crosstab(filtered_df['BORO'], filtered_df['GRADE'], normalize='index') * 100
        fig_boro = go.Figure()
        for grade in ['A', 'B', 'C']:
            if grade in boro_grade.columns:
                fig_boro.add_trace(go.Bar(
                    name=grade,
                    x=boro_grade.index,
                    y=boro_grade[grade],
                    marker_color={'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}[grade]
                ))
        fig_boro.update_layout(
            title="Grade Distribution by Borough (%)",
            barmode='group',
            height=400,
            xaxis_title="Borough",
            yaxis_title="Percentage"
        )
        st.plotly_chart(fig_boro, use_container_width=True)
    
    with col2:
        # Violations by grade
        fig_box = go.Figure()
        for grade in ['A', 'B', 'C']:
            grade_data = filtered_df[filtered_df['GRADE'] == grade]
            fig_box.add_trace(go.Box(
                y=grade_data['TOTAL_VIOLATIONS'],
                name=grade,
                marker_color={'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}[grade]
            ))
        fig_box.update_layout(
            title="Total Violations by Grade",
            height=400,
            yaxis_title="Total Violations"
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.header("Explore the Data")
    
    # Time series
    st.subheader("Inspection Trends Over Time")
    monthly = filtered_df.groupby(filtered_df['INSPECTION_DATE'].dt.to_period('M')).agg({
        'SCORE': 'mean',
        'CAMIS': 'count'
    }).reset_index()
    monthly['INSPECTION_DATE'] = monthly['INSPECTION_DATE'].dt.to_timestamp()
    
    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_time.add_trace(
        go.Scatter(x=monthly['INSPECTION_DATE'], y=monthly['SCORE'], 
                   name="Avg Score", line=dict(color='#e74c3c', width=3)),
        secondary_y=False
    )
    fig_time.add_trace(
        go.Bar(x=monthly['INSPECTION_DATE'], y=monthly['CAMIS'], 
               name="Inspection Count", marker_color='#3498db', opacity=0.3),
        secondary_y=True
    )
    fig_time.update_xaxes(title_text="Date")
    fig_time.update_yaxes(title_text="Average Score", secondary_y=False)
    fig_time.update_yaxes(title_text="Inspection Count", secondary_y=True)
    fig_time.update_layout(height=400, title="Inspection Trends")
    st.plotly_chart(fig_time, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top cuisines by score
        st.subheader("Cuisines with Highest Scores")
        cuisine_stats = filtered_df.groupby('CUISINE').agg({
            'SCORE': 'mean',
            'CAMIS': 'count'
        }).reset_index()
        cuisine_stats = cuisine_stats[cuisine_stats['CAMIS'] >= 20]
        top_10 = cuisine_stats.nlargest(10, 'SCORE')
        
        fig_cuisine = px.bar(
            top_10,
            x='SCORE',
            y='CUISINE',
            orientation='h',
            color='SCORE',
            color_continuous_scale='Reds',
            title="Top 10 Cuisines (min 20 inspections)"
        )
        fig_cuisine.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_cuisine, use_container_width=True)
    
    with col2:
        # Violation patterns
        st.subheader("Violation Patterns")
        
        # Critical vs total violations scatter
        sample_df = filtered_df.sample(min(5000, len(filtered_df)))
        fig_scatter = px.scatter(
            sample_df,
            x='TOTAL_VIOLATIONS',
            y='CRITICAL_VIOLATIONS',
            color='GRADE',
            color_discrete_map={'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'},
            opacity=0.6,
            title="Critical vs Total Violations"
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Data table
    st.subheader("Raw Data Sample")
    display_cols = ['RESTAURANT_NAME', 'BORO', 'CUISINE', 'GRADE', 'SCORE', 
                    'TOTAL_VIOLATIONS', 'CRITICAL_VIOLATIONS', 'INSPECTION_DATE']
    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True,
        height=400
    )

with tab3:
    st.header("Model Performance")
    
    st.markdown("""
    Two models were trained to predict restaurant grades:
    - **Logistic Regression** (baseline)
    - **Random Forest** (best model)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "93.8%")
    with col2:
        st.metric("A Grade Recall", "99%")
    with col3:
        st.metric("Test Set Size", "10,368")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        st.subheader("Confusion Matrix (Random Forest)")
        cm_data = [
            [8876, 111, 16],
            [244, 575, 76],
            [83, 112, 275]
        ]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Predicted A', 'Predicted B', 'Predicted C'],
            y=['Actual A', 'Actual B', 'Actual C'],
            colorscale='Greens',
            text=cm_data,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        fig_cm.update_layout(height=400, title="")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.info("**Note:** Model is excellent at predicting A grades (99% recall) but struggles with B and C grades due to class imbalance.")
    
    with col2:
        # Feature importance
        st.subheader("Feature Importance")
        features = ['Critical Violations', 'Total Violations', 'Cuisine Type', 
                    'Month', 'Day of Week', 'Borough']
        importances = [44.4, 30.5, 9.9, 6.6, 4.8, 3.9]
        
        fig_imp = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Blues',
                showscale=False
            ),
            text=[f'{x}%' for x in importances],
            textposition='auto'
        ))
        fig_imp.update_layout(
            height=400,
            xaxis_title="Importance (%)",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.success("**Key Insight:** Critical violations alone account for 44% of the model's predictions. Food safety is the main driver of grades.")
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("Model Comparison")
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [93.75, 93.81],
        'A Recall': [100, 99],
        'B Recall': [57, 64],
        'C Recall': [51, 59]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    fig_comp = go.Figure()
    for col in ['A Recall', 'B Recall', 'C Recall']:
        fig_comp.add_trace(go.Bar(
            name=col,
            x=comparison_df['Model'],
            y=comparison_df[col],
            text=comparison_df[col],
            textposition='auto',
            texttemplate='%{text}%'
        ))
    fig_comp.update_layout(
        title="Model Recall by Grade",
        barmode='group',
        yaxis_title="Recall (%)",
        height=400
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with tab4:
    st.header("Predict Restaurant Grade")
    
    if model is None or cuisine_encoder is None or boro_encoder is None:
        st.error("Model files not found. Please ensure random_forest_model.pkl and encoder files are in the models/ folder.")
    else:
        st.markdown("Enter restaurant inspection details to predict the grade:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_viol = st.number_input("Total Violations", min_value=0, max_value=20, value=2)
            critical_viol = st.number_input("Critical Violations", min_value=0, max_value=20, value=1)
        
        with col2:
            month = st.selectbox("Month", list(range(1, 13)), index=5)
            day_of_week = st.selectbox("Day of Week", 
                                       ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                        'Friday', 'Saturday', 'Sunday'], index=2)
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                       'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            day_encoded = day_map[day_of_week]
        
        with col3:
            cuisine_input = st.selectbox("Cuisine", sorted(df['CUISINE'].dropna().unique()))
            boro_input = st.selectbox("Borough", sorted(df['BORO'].dropna().unique()))
        
        if st.button("🎯 Predict Grade", type="primary"):
            try:
                # Encode inputs using saved encoders
                cuisine_encoded = cuisine_encoder.transform([cuisine_input])[0]
                boro_encoded = boro_encoder.transform([boro_input])[0]
                
                # Make prediction
                features = [[total_viol, critical_viol, month, day_encoded, cuisine_encoded, boro_encoded]]
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"### Predicted Grade: **{prediction}**")
                    if prediction == 'A':
                        st.success("Excellent!")
                    elif prediction == 'B':
                        st.warning("Needs Improvement")
                    else:
                        st.error("Critical Issues")
                
                with col2:
                    st.markdown("### Confidence:")
                    max_prob = max(probabilities)
                    st.progress(max_prob)
                    st.markdown(f"**{max_prob*100:.1f}%** confidence")
                
                with col3:
                    st.markdown("### Probabilities:")
                    for i, grade in enumerate(['A', 'B', 'C']):
                        st.markdown(f"**{grade}:** {probabilities[i]*100:.1f}%")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Make sure the cuisine and borough values match those in the training data.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>NYC Restaurant Inspection ML Dashboard</strong></p>
</div>
""", unsafe_allow_html=True)

