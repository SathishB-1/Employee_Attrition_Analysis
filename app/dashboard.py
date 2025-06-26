import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import joblib

# Load data and model
df = pd.read_csv('data/HREmployee_data.csv')
model = joblib.load('models/attrition_model.pkl')

# Setup app
app = dash.Dash(__name__)
app.title = 'HR Attrition Dashboard'

# Layout
app.layout = html.Div([
    html.H1("Employee Attrition Dashboard"),
    
    html.Div([
        html.Label("Select Department:"),
        dcc.Dropdown(
            options=[{'label': d, 'value': d} for d in df['Department'].unique()],
            id='department-dropdown',
            value=df['Department'].unique()[0]
        ),
    ]),

    html.Div([
        dcc.Graph(id='attrition-rate-plot')
    ]),

    html.Div([
        dcc.Graph(id='age-salary-plot')
    ]),
])

# Callbacks
@app.callback(
    Output('attrition-rate-plot', 'figure'),
    Input('department-dropdown', 'value')
)
def update_attrition_plot(dept):
    filtered_df = df[df['Department'] == dept]
    print(f"Selected department: {dept}")
    print(f"Filtered rows: {len(filtered_df)}")

    if filtered_df.empty:
        return go.Figure()  # Empty plot

    attrition_counts = filtered_df['Attrition'].value_counts()
    fig = px.bar(x=attrition_counts.index, y=attrition_counts.values, labels={'x': 'Attrition', 'y': 'Count'})
    return fig


@app.callback(
    Output('age-salary-plot', 'figure'),
    Input('department-dropdown', 'value')
)
def update_age_salary_plot(dept):
    filtered_df = df[df['Department'] == dept]
    if filtered_df.empty:
        return go.Figure()

    fig = px.scatter(
    filtered_df,
    x='Age',
    y='DailyRate',
    color='Attrition',
    title=f'Age vs Daily Rate for {dept} Department'
)

    return fig

if __name__ == '__main__':
    app.run(debug=True)

