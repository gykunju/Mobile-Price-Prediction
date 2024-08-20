# Mobile Price Prediction
## Project Overview
This project is a machine learning application designed to predict the price range of mobile devices based on various hardware and software features. The data for this project is sourced from Kaggle's Mobile Price Classification dataset. The goal is to provide a predictive model that can categorize mobile devices into one of four price ranges: Low Cost, Medium Cost, High Cost, and Very High Cost.

## Table of Contents
- Project Overview
- Dataset
- Exploratory Data Analysis (EDA)
- Modeling
- Prediction Interface
- Usage
- Dataset
  
The dataset used in this project contains 21 features related to mobile devices, such as battery power, RAM, and screen dimensions, and a target variable price_range which categorizes the devices into four classes (0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost).

## Features
- battery_power: Total energy a battery can store in one time measured in mAh.
- blue: Has Bluetooth or not.
- clock_speed: Speed at which microprocessor executes instructions.
- dual_sim: Has dual sim support or not.
- fc: Front camera megapixels.
- four_g: Has 4G or not.
- int_memory: Internal memory in GB.
- m_dep: Mobile depth in cm.
- mobile_wt: Weight of the mobile phone in grams.
- pc: Primary camera megapixels.
- px_height: Pixel resolution height.
- px_width: Pixel resolution width.
- ram: Random access memory in MB.
- sc_h: Screen height of the mobile in cm.
- sc_w: Screen width of the mobile in cm.
- talk_time: Longest time that a single battery charge will last when you are constantly talking on the phone.
- three_g: Has 3G or not.
- touch_screen: Has a touch screen or not.
- wifi: Has WiFi or not.
- n_cores: Number of cores of the processor.


## Exploratory Data Analysis (EDA)
The EDA was conducted to understand the distribution of data, the relationship between features, and the significance of each feature in predicting the target variable.

### Key Insights
The data is evenly distributed across the four price ranges.

ram shows a strong correlation with price_range, indicating that higher RAM often corresponds to a higher price range.

Features like px_width and px_height also show a significant correlation with price_range.

Some features, such as battery_power and talk_time, do not show a direct correlation with each other or with price_range.

Visualizations used in the EDA include:

Pie Chart: To show the distribution of the target variable.

Correlation Heatmap: To show the relationship between different features and the target.

Boxplots: To visualize the distribution of continuous features.

Scatter Plots: To analyze the relationships between pairs of features.

## Modeling
### Model Selection
Two models were chosen for this project:

### - Logistic Regression
### - Support Vector Classifier (SVC)

Both models performed well, with Logistic Regression having slightly better accuracy on the training data. A Stacking Ensemble was then used, combining these models to further improve prediction accuracy.

### Stacking Ensemble
Estimators: SVC and Logistic Regression.
Final Estimator: Logistic Regression.

## Model Performance
Logistic Regression: Train Accuracy = 1.00, Test Accuracy = 0.97.
SVC: Train Accuracy = 0.97, Test Accuracy = 0.96.
Stacking Model: Train Accuracy = 0.98, Test Accuracy = 0.97.

## Prediction Interface
A user-friendly interface was built using Streamlit to allow users to input the characteristics of their mobile device and get a predicted price range.

### Features of the Interface
Users can input values for features such as battery power, RAM, screen size, etc.
The model predicts the price range and displays the result(range) on the interface
