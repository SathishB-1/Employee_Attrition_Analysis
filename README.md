## Employee Attrition Prediction Dashboard:

This is a web-based dashboard built using Plotly Dash to visualize and analyze employee attrition patterns across departments. It helps HR professionals and managers identify potential risk areas and trends contributing to employee turnover.

## Objective:

To create an interactive dashboard that visualizes employee attrition data using various metrics such as age, salary, department, job satisfaction, and more, aiding data-driven decision-making in talent management.

## Features:

Interactive department filter

Real-time updates of attrition-related charts

Visualizations for attrition rate, age vs. income, and other key HR metrics

Callback error handling and modular layout

Clean, responsive user interface using Plotly Dash

## Technologies Used:

Python 

Pandas and NumPy for data processing

Plotly Express and Dash for interactive visualization

Dash Bootstrap Components for UI styling

Jupyter Notebook for EDA and preprocessing

CSV as the input dataset format

## How to Run Locally:

Clone this repository:
git clone https://github.com/your-username/employee-attrition-dashboard.git

cd employee-attrition-dashboard

## Install dependencies:

pip install -r requirements.txt

## Run the Dash app:

python app.py

## Input Dataset:

The dataset should include the following columns (at minimum):

Age

Attrition (Yes/No)

Department

JobRole

MonthlyIncome

JobSatisfaction

YearsAtCompany

Other relevant HR metrics

## Main Visualizations:

Attrition Count by Department

Age vs Monthly Income for employees (to explore salary distribution)

Attrition distribution over age or experience

Satisfaction or performance metrics vs. attrition

## Error Handling:

The dashboard includes callback error logging. Common errors include:

Missing column names in CSV file

Wrongly typed or unexpected input features

Mismatched feature names in update functions

## Machine Learning (Optional):
While the current version focuses on visualization, this dashboard can be extended to include machine learning models (e.g., RandomForest or decision trees) to predict attrition probability.

## File Structure:

dashborad.py: Main Dash application script

data/: Input data CSV file

requirements.txt: All dependencies

README.txt: Project overview (this file)

## License:

This project is open source under the MIT License.

## Dashboard Interface:

![img Alt]()