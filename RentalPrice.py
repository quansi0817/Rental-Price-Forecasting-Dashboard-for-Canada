import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px


def clean_beds(value):
    if pd.isna(value):
        return np.nan
    value = str(value).lower()
    if 'studio' in value:
        return 0
    elif 'none' in value:
        return 0
    else:
        try:
            return int(''.join(filter(str.isdigit, value)))
        except ValueError:
            return np.nan

def clean_property_type(value):
    valid_types = ['Townhouse', 'Apartment', 'Basement', 'House', 'Condo Unit']
    if pd.isna(value):
        return 'Other'
    value = str(value).strip().title()
    return value if value in valid_types else 'Other'
def load_data():
    try:
        df = pd.read_csv('rentfaster.csv')
        print("Columns in the CSV file:", df.columns.tolist())
        print(f"Initial shape of the dataframe: {df.shape}")
        
        # 确保 'City' 列包含省份信息
        df['city'] = df.apply(lambda row: f"{row['city']}, {row['province']}" if 'province' in df.columns else row['city'], axis=1)
        
        # 将所有列名转换为小写
        df.columns = df.columns.str.lower()
        
        column_mapping = {
            'city': 'City',
            'type': 'PropertyType',
            'price': 'Price',
            'beds': 'Bedrooms',
            'sq_feet': 'SquareFeet',
            'cats': 'Cats',
            'dogs': 'Dogs'
        }
        df = df.rename(columns=column_mapping)
        
        required_columns = ['City', 'PropertyType', 'Price', 'Bedrooms', 'SquareFeet', 'Cats', 'Dogs']
        df = df[required_columns]
        
        df['Bedrooms'] = df['Bedrooms'].apply(clean_beds)
        df['PropertyType'] = df['PropertyType'].apply(clean_property_type)
        
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['SquareFeet'] = pd.to_numeric(df['SquareFeet'], errors='coerce')
        
        # Filter out entries with Price less than 500
        df = df[df['Price'] >= 500]
        df = df[df['SquareFeet'] >= 300]

        # Convert 'Cats' and 'Dogs' to binary
        df['Cats'] = df['Cats'].astype(bool).astype(int)
        df['Dogs'] = df['Dogs'].astype(bool).astype(int)
        
        # Remove rows with NaN in Price or SquareFeet
        df = df.dropna(subset=['Price', 'SquareFeet'])
        
        # Fill NaN in Bedrooms with median
        df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())

        print(f"Shape after cleaning: {df.shape}")
        print("Column types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        
        return df
    
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise


def train_model(df, target='Price'):
    try:
        features = df.columns.drop([target])
        numeric_features = ['Bedrooms', 'SquareFeet', 'Price']
        numeric_features = [f for f in numeric_features if f != target]
        categorical_features = ['City', 'PropertyType']
        boolean_features = ['Cats', 'Dogs']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        boolean_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('bool', boolean_transformer, boolean_features)
            ])
        
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        
        X = df.drop(target, axis=1)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Model R-squared score on training data ({target}): {train_score:.3f}")
        print(f"Model R-squared score on test data ({target}): {test_score:.3f}")
        
        return model, X.columns.tolist()
    except Exception as e:
        print(f"An error occurred in train_model: {str(e)}")
        raise


def predict(model, feature_columns, province, city, property_type, bedrooms, square_feet, cats, dogs):
    input_data = pd.DataFrame({
        'City': [f"{city}, {province}"],
        'PropertyType': [property_type],
        'Bedrooms': [bedrooms],
        'SquareFeet': [square_feet],
        'Price': [0],  # 添加一个虚拟的Price值，在预测SquareFeet时会被忽略
        'Cats': [int(cats)],
        'Dogs': [int(dogs)]
    })
    
    prediction = model.predict(input_data)
    return prediction[0]



# 样式定义
styles = {
    'container': {
        'max-width': '1200px',
        'margin': '0 auto',
        'padding': '20px',
        'font-family': 'Arial, sans-serif'
    },
    'header': {
        'text-align': 'center',
        'color': '#2c3e50',
        'margin-bottom': '30px'
    },
    'input-container': {
        'margin-bottom': '20px'
    },
    'label': {
        'font-weight': 'bold',
        'margin-bottom': '5px',
        'display': 'block'
    },
    'dropdown': {
        'width': '100%',
        'margin-bottom': '10px'
    },
    'checklist': {
        'margin-bottom': '20px'
    },
    'button': {
        'background-color': '#3498db',
        'color': 'white',
        'padding': '10px 20px',
        'border': 'none',
        'cursor': 'pointer',
        'font-size': '16px',
        'margin-top': '20px'
    },
    'output': {
        'margin-top': '30px',
        'font-size': '18px',
        'font-weight': 'bold',
        'text-align': 'center'
    },
    'graph-container': {
        'display': 'flex',
        'justify-content': 'space-between',
        'margin-top': '30px'
    },
    'graph': {
        'width': '48%'
    }
}


# 加载数据和训练模型
df = load_data()
price_model, price_feature_columns = train_model(df, target='Price')
area_model, area_feature_columns = train_model(df, target='SquareFeet')



# 获取唯一的省份和城市
provinces = sorted(df['City'].str.split(', ').str[-1].unique())
cities_by_province = {province: sorted(df[df['City'].str.endswith(province)]['City'].str.split(', ').str[0].unique()) for province in provinces}

# 创建 Dash 应用
app = dash.Dash(__name__)

# 应用布局
app.layout = html.Div(style=styles['container'], children=[
    html.H1("Canada Rental Price Predictor", style=styles['header']),
    
    html.Div(style=styles['input-container'], children=[
        html.Label("Province:", style=styles['label']),
        dcc.Dropdown(
            id='province-dropdown',
            options=[{'label': province, 'value': province} for province in provinces],
            value=provinces[0],
            style=styles['dropdown']
        ),
    ]),
    
    html.Div(style=styles['input-container'], children=[
        html.Label("City:", style=styles['label']),
        dcc.Dropdown(id='city-dropdown', style=styles['dropdown'])
    ]),
    
    html.Div(style=styles['input-container'], children=[
        html.Label("Property Type:", style=styles['label']),
        dcc.Dropdown(
            id='property-type-dropdown',
            options=[{'label': prop, 'value': prop} for prop in df['PropertyType'].unique()],
            value=df['PropertyType'].iloc[0],
            style=styles['dropdown']
        ),
    ]),
    
    html.Div(style=styles['input-container'], children=[
        html.Label("Number of Bedrooms:", style=styles['label']),
        dcc.Dropdown(
            id='bedrooms-dropdown',
            options=[{'label': f'{bed} Bedroom(s)', 'value': bed} for bed in sorted(df['Bedrooms'].unique())],
            value=df['Bedrooms'].iloc[0],
            style=styles['dropdown']
        ),
    ]),
    
    html.Div(style=styles['checklist'], children=[
        html.Label("Pet Friendly:", style=styles['label']),
        dcc.Checklist(
            id='pets-checklist',
            options=[
                {'label': 'Cats Allowed', 'value': 'cats'},
                {'label': 'Dogs Allowed', 'value': 'dogs'}
            ],
            value=[]
        ),
    ]),
    
    html.Button('Predict', id='predict-button', n_clicks=0, style=styles['button']),
    
    html.Div(id='prediction-output', style=styles['output']),
    
    html.Div(id='graph-container', style=styles['graph-container'], children=[
        html.Div(id='price-distribution-graph', style=styles['graph']),
        html.Div(id='area-distribution-graph', style=styles['graph'])
    ])
])

@app.callback(
    Output('city-dropdown', 'options'),
    [Input('province-dropdown', 'value')]
)
def update_cities(selected_province):
    return [{'label': city, 'value': city} for city in cities_by_province[selected_province]]

@app.callback(
    [Output('prediction-output', 'children'),
     Output('price-distribution-graph', 'children'),
     Output('area-distribution-graph', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('province-dropdown', 'value'),
     State('city-dropdown', 'value'),
     State('property-type-dropdown', 'value'),
     State('bedrooms-dropdown', 'value'),
     State('pets-checklist', 'value')]
)
def update_prediction_and_graphs(n_clicks, province, city, property_type, bedrooms, pets):
    if n_clicks > 0:
        cats = 'cats' in pets
        dogs = 'dogs' in pets
        
        try:
            # 使用平均SquareFeet进行初始价格预测
            avg_square_feet = df['SquareFeet'].mean()
            predicted_price = predict(price_model, price_feature_columns, province, city, property_type, bedrooms, avg_square_feet, cats, dogs)
            
            # 使用预测的价格来预测SquareFeet
            predicted_area = predict(area_model, area_feature_columns, province, city, property_type, bedrooms, predicted_price, cats, dogs)
            
            # 使用预测的SquareFeet重新预测价格以获得更准确的结果
            final_predicted_price = predict(price_model, price_feature_columns, province, city, property_type, bedrooms, predicted_area, cats, dogs)
            
            # Filter data for the selected province
            filtered_df = df[df['City'].str.endswith(province)]
            
            # Create price distribution graph for the province
            price_fig = px.box(filtered_df, x='PropertyType', y='Price', 
                               title=f'Price Distribution in {province} by Property Type',
                               labels={'Price': 'Rental Price ($)', 'PropertyType': 'Property Type'})
            price_fig.add_hline(y=final_predicted_price, line_dash="dash", line_color="red", annotation_text="Predicted Price")
            
            # Create area distribution graph for the province
            area_fig = px.box(filtered_df, x='PropertyType', y='SquareFeet', 
                              title=f'Area Distribution in {province} by Property Type',
                              labels={'SquareFeet': 'Area (sq ft)', 'PropertyType': 'Property Type'})
            area_fig.add_hline(y=predicted_area, line_dash="dash", line_color="red", annotation_text="Predicted Area")
            
            return (
                f"Predicted Rental Price: ${final_predicted_price:.2f}\nPredicted Area: {predicted_area:.2f} sq ft",
                dcc.Graph(figure=price_fig),
                dcc.Graph(figure=area_fig)
            )
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return f"Error in prediction: {str(e)}", None, None
    return "Please click the 'Predict' button to see the prediction and analysis.", None, None

if __name__ == '__main__':
    app.run_server(debug=True)