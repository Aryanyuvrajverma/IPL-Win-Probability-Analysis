# IPL-Win-Probability-Analysis
End-to-end IPL data analytics project with win probability prediction using machine learning and interactive Power BI dashboards. Includes feature engineering, EDA, and match outcome modeling.

## Dashboard Preview

<img width="1278" height="718" alt="Screenshot 2026-04-09 120905" src="https://github.com/user-attachments/assets/2005a773-7e55-4b79-ae7c-7a0762a46ed2" />


Overview

This project presents an end-to-end analysis of IPL (Indian Premier League) match data, aimed at uncovering key performance patterns and understanding the factors influencing match outcomes.

The analysis combines exploratory data analysis (EDA), feature engineering, and machine learning techniques to evaluate how match context impacts winning probabilities. Additionally, an interactive dashboard has been developed to support data-driven insights and visualization.

Dataset
The project utilizes IPL ball-by-ball and match-level data.

Key statistics:
Matches analyzed: 1000+
Total runs: 374,000+
Total wickets: 14,000+

Data includes:
Match details (teams, venue, result)
Ball-by-ball events (runs, wickets, overs)
Contextual match variables (run rate, balls remaining, wickets remaining)

Methodology
Data Preprocessing
Standardized column names and formats
Handled missing values and removed duplicates
Created derived variables such as total runs and wicket indicators
Feature Engineering
Pressure Index (required run rate vs current run rate)
Run rate difference
Overs completed and balls used
Wickets lost and remaining
Modeling

Machine learning models were developed to evaluate the influence of match conditions on outcomes:
Random Forest Classifier
Logistic Regression

##These models help in understanding how contextual match features contribute to predicting winning scenarios.

Key Insights
Wickets remaining are the most significant predictor of match outcomes
Death overs represent the most critical and high-impact phase
Teams exhibit a strong preference for chasing (fielding first)
Match context variables have a greater influence than team or venue alone

Tools & Technologies
-->Python
-->Pandas 
-->NumPy
-->Scikit-learn
-->Matplotlib
-->Power BI

Execution Steps
Clone the repository:
git clone https://github.com/yourusername/your-repo-name.git

Install required dependencies:
pip install -r requirements.txt

Run the main script:
python data_cleaning.py

Open the Power BI dashboard for visualization insights

Future Enhancements
Implementation of advanced models (e.g., XGBoost, Gradient Boosting)
Development of an interactive application for real-time predictions
Continuous addition of new insights and feature improvements
Implementation of advanced models (e.g., XGBoost, Gradient Boosting)
Development of an interactive application for real-time predictions
Continuous addition of new insights and feature improvements
