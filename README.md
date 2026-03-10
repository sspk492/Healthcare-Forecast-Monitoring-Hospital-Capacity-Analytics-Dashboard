# Healthcare Forecast Monitoring & Hospital Capacity Analytics Dashboard

This project develops a healthcare operations analytics and forecasting monitoring system using U.S. CDC hospital capacity time-series data. The goal is to analyze hospital admission trends, monitor healthcare system capacity, and evaluate the performance of forecasting models used to predict patient demand.

Hospitals rely heavily on accurate forecasts of patient admissions to plan staffing levels, manage ICU capacity, and allocate resources efficiently. However, forecasting models can degrade over time due to seasonal changes or unexpected health events. This project builds a monitoring framework that compares actual hospital admissions with predicted values and tracks forecasting errors.

---

# Project Objectives

- Analyze national hospital admission trends
- Forecast hospital admissions using time-series modeling
- Monitor forecasting accuracy using MAE, RMSE, and MAPE
- Identify healthcare capacity pressure indicators
- Analyze demographic patterns in hospital admissions
- Build interactive dashboards for operational monitoring

---

# Dataset

**Source:** U.S. CDC COVID-19 Reported Patient Impact and Hospital Capacity Data

The dataset includes:

- State-level hospital admissions
- ICU bed utilization
- Inpatient bed occupancy
- COVID-related deaths
- Demographic admission counts

Time Range: 2020 – 2024


---

# Tools & Technologies

Python  
Pandas  
Scikit-Learn  
Time-Series Analysis  
Tableau  
Data Visualization  

---

## Project Workflow

→ Data Extraction  
→ Data Cleaning & Feature Engineering  
→ Exploratory Data Analysis  
→ Forecast Model Development  
→ Forecast Evaluation  
→ Forecast Monitoring Dataset  
→ Interactive Tableau Dashboards


---

# Dashboards

## Executive Healthcare Overview

Key metrics monitored:

- Total hospital admissions
- ICU utilization rate
- Bed occupancy rate
- COVID-related deaths
- Geographic distribution of admissions

---

## Forecast Monitoring

Forecast monitoring tracks:

- Actual vs predicted admissions
- Forecast error trends
- Model performance metrics

Model Performance:
- MAE : 118.8
- RMSE : 161.1
- MAPE : 4.73%


---

## Demographic Insights

Demographic analysis reveals:

- Hospital admissions increase significantly for patients aged **60+**
- Age groups **70–79 and 80+** show the highest hospitalization rates
- Younger age groups represent a smaller share of admissions
- Seasonal spikes impact all age groups but disproportionately affect older populations

---

# Key Insights

- Hospital admissions peaked during major COVID waves between **2020 and 2022**
- ICU utilization averaged **~70%**, approaching critical capacity levels
- California, Texas, and Florida recorded the highest cumulative admissions
- The forecasting model tracks admission trends closely and outperforms a baseline approach


---

# Tableau Dashboard

The Tableau workbook is available in:
tableau workbook/Health_Care_Dashboard.twbx

This dashboard provides an interactive interface to monitor hospital demand, forecasting performance, and demographic admission patterns.

---

# Author

**Sankar Prudhvi Krishna Simhadri**

Master’s in Data Analytics.

