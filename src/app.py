import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport

import datetime
import plotly
import iplot

import plotly.offline as pyoff
import plotly.graph_objs as go

from plotly.subplots import make_subplots

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from mlforecast import MLForecast
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean

####Données & Manipulations####
traffic_df = pd.read_parquet('src/data/traffic_10lines.parquet')

(traffic_df
 .groupby(['home_airport', 'paired_airport', 'direction'])
 .agg(date_min=('date', 'min'), date_max=('date', 'max'), pax=('pax', 'sum'))
 .reset_index()
)

def generate_route_df(traffic_df: pd.DataFrame, homeAirport: str, pairedAirport: str) -> pd.DataFrame:
  """Extract route dataframe from traffic dataframe for route from home airport to paired airport

  Args:
  - traffic_df (pd.DataFrame): traffic dataframe
  - homeAirport (str): IATA Code for home airport
  - pairedAirport (str): IATA Code for paired airport

  Returns:
  - pd.DataFrame: aggregated daily PAX traffic on route (home-paired)
  """
  _df = (traffic_df
         .query('home_airport == "{home}" and paired_airport == "{paired}"'.format(home=homeAirport, paired=pairedAirport))
         .groupby(['home_airport', 'paired_airport', 'date'])
         .agg(pax_total=('pax', 'sum'))
         .reset_index()
         )
  return _df

def draw_ts_multiple(df: pd.DataFrame, v1: str, v2: str=None, prediction: str=None, date: str='date',
              secondary_y=True, covid_zone=False, display=True):
  """Draw times series possibly on two y axis, with COVID period option.

  Args:
  - df (pd.DataFrame): time series dataframe (one line per date, series in columns)
  - v1 (str | list[str]): name or list of names of the series to plot on the first x axis
  - v2 (str): name of the serie to plot on the second y axis (default: None)
  - prediction (str): name of v1 hat (prediction) displayed with a dotted line (default: None)
  - date (str): name of date column for time (default: 'date')
  - secondary_y (bool): use a secondary y axis if v2 is used (default: True)
  - covid_zone (bool): highlight COVID-19 period with a grayed rectangle (default: False)
  - display (bool): display figure otherwise just return the figure (default: True)

  Returns:
  - fig (plotly.graph_objs._figure.Figure): Plotly figure generated

  Notes:
  Make sure to use the semi-colon trick if you don't want to have the figure displayed twice.
  Or use `display=False`.
  """
  if isinstance(v1, str):
    variables = [(v1, 'V1')]
  else:
    variables = [(v, 'V1.{}'.format(i)) for i, v in enumerate(v1)]
  title = '<br>'.join([n + ': '+ v for v, n in variables]) + ('<br>V2: ' + v2) if v2 else '<br>'.join([v + ': '+ n for v, n in variables])
  layout = dict(
    title=title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
  )
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.update_layout(layout)
  for v, name in variables:
    fig.add_trace(go.Scatter(x=df[date], y=df[v], name=name), secondary_y=False)
  if v2:
    fig.add_trace(go.Scatter(x=df[date], y=df[v2], name='V2'), secondary_y=secondary_y)
    fig['layout']['yaxis2']['showgrid'] = False
    fig.update_yaxes(rangemode='tozero')
    fig.update_layout(margin=dict(t=125 + 30 * (len(variables) - 1)))
  if prediction:
    fig.add_trace(go.Scatter(x=df[date], y=df[prediction], name='^V1', line={'dash': 'dot'}), secondary_y=False)

  if covid_zone:
    fig.add_vrect(
        x0=pd.Timestamp("2020-03-01"), x1=pd.Timestamp("2022-01-01"),
        fillcolor="Gray", opacity=0.5,
        layer="below", line_width=0,
    )
  if display:
    pyoff.iplot(fig)
  return fig

routes = (traffic_df
 .drop_duplicates(subset=['home_airport', 'paired_airport'])
 [['home_airport', 'paired_airport']]
 .to_dict(orient='rows')
)


####Paramètres####
st.title('Traffic Forecaster')

route = routes
options = ['{home} to {paired}'.format(home=entry['home_airport'], paired=entry['paired_airport']) for entry in route]
#HOME_AIRPORTS = ("LGW", "LIS", "LYS")
#PAIRED_AIRPORTS = ("FUE", "AMS", "ORY")
model_list = ("Prophet", "LGBMRegressor", "XGBRegressor", "RandomForestRegressor")

with st.sidebar:
    #home_airport = st.selectbox(
        #'Home Airport', HOME_AIRPORTS)
    #paired_airport = st.selectbox(
        #'Paired Airport', PAIRED_AIRPORTS)
    route_selection = st.selectbox( 
        'Route selected', options)
    #forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 7, 365, 300)
    MODELS = st.selectbox('Model used to forecast', model_list)
    run_forecast = st.button('Forecast')

st.write('Home airport of the route :', route_selection.split(' to ')[0])
st.write('Paired airport of the route :', route_selection.split(' to ')[1])
#st.write('Home Airport selected:', home_airport)
#st.write('Paired Airport selected:', paired_airport)
#st.write('Date selected:', forecast_date)
st.write('Days of forecast', nb_days)

#st.write(df.query('home_airport = "{}"'.format(home_airport)).shape[0]) 

####Graph et Modeles####
home = route_selection.split(' to ')[0]
paired= route_selection.split(' to ')[1]
data = generate_route_df(traffic_df, home, paired)
         
if run_forecast and MODELS == "Prophet":
    #Prophet
    models_prophet = dict()
    performances = dict()
    # Build route traffic dataframe
    _df = generate_route_df(traffic_df, home, paired)
    # Create a model
    _model = Prophet()
    # Fit the model
    _model.fit(_df.rename(columns={'date': 'ds', 'pax_total': 'y'}))
    # Cross validate the model
    _cv_df = cross_validation(_model, horizon='90 days', parallel="processes")
    _perf_df = performance_metrics(_cv_df, rolling_window=1)
    # Save results
    models_prophet[(home, paired)] = _model
    performances[home, paired] = _perf_df['rmse'].values[0]
    
    #create the graph of the forecast : 
    future = models_prophet[(home, paired)].make_future_dataframe(periods=nb_days)
    forecast = models_prophet[(home, paired)].predict(future)
    forecast = forecast[['ds', 'yhat']].tail(nb_days).rename(columns={"ds": "date", "yhat": "Forecasts"})
    fig = draw_ts_multiple(pd.concat([data[['home_airport', 'date', 'pax_total']]
                                      .rename(columns={"pax_total": "Passengers"}),forecast]),
                   v1='Passengers', v2='Forecasts', secondary_y=False, covid_zone=True)
    st.write(fig)

elif run_forecast and MODELS == "LGBMRegressor":
    tested_models = [
        lgb.LGBMRegressor(),
        ]

    @njit
    def rolling_mean_28(x):
        return rolling_mean(x, window_size=28)

    fcst = MLForecast(
        models=tested_models,
        freq='D',
        lags=[7, 14, 21, 28],
        lag_transforms={
            1: [expanding_mean],
            7: [rolling_mean_28]
        },
        date_features=['dayofweek'],
        differences=[1],
    )
    nixtla_model = fcst.fit(generate_route_df(traffic_df, home, paired).drop(columns=['paired_airport']),
                    id_col='home_airport', time_col='date', target_col='pax_total')
    fig = draw_ts_multiple((pd.concat([generate_route_df(traffic_df, home, paired)
                                       .drop(columns=['paired_airport'])
                                       .rename(columns={"pax_total": "Passengers"}),nixtla_model.predict(nb_days)])), 
                           v1='Passengers', v2='LGBMRegressor', secondary_y=False, covid_zone=True);
    st.write(fig)
        
elif run_forecast and MODELS == "XGBRegressor": 
    tested_models = [
        xgb.XGBRegressor(),
    ]

    @njit
    def rolling_mean_28(x):
        return rolling_mean(x, window_size=28)
    
    fcst = MLForecast(
        models=tested_models,
        freq='D',
        lags=[7, 14, 21, 28],
        lag_transforms={
            1: [expanding_mean],
            7: [rolling_mean_28]
        },
        date_features=['dayofweek'],
        differences=[1],
    )
    nixtla_model = fcst.fit(generate_route_df(traffic_df, home, paired).drop(columns=['paired_airport']),
                    id_col='home_airport', time_col='date', target_col='pax_total')
    fig = draw_ts_multiple((pd.concat([generate_route_df(traffic_df, home, paired)
                                       .drop(columns=['paired_airport'])
                                       .rename(columns={"pax_total": "Passengers"}),
                                 nixtla_model.predict(nb_days)])),
                     v1='Passengers', v2='XGBRegressor', secondary_y=False, covid_zone=True); 
    st.write(fig)
         
elif run_forecast and MODELS == "RandomForestRegressor": 
    tested_models = [
        RandomForestRegressor(random_state=0),
    ]
    @njit
    def rolling_mean_28(x):
        return rolling_mean(x, window_size=28)
    fcst = MLForecast(
        models=tested_models,
        freq='D',
        lags=[7, 14, 21, 28],
        lag_transforms={
            1: [expanding_mean],
            7: [rolling_mean_28]
        },
        date_features=['dayofweek'],
        differences=[1],
    )
    nixtla_model = fcst.fit(generate_route_df(traffic_df, home, paired).drop(columns=['paired_airport']),
                    id_col='home_airport', time_col='date', target_col='pax_total')
    fig = draw_ts_multiple((pd.concat([generate_route_df(traffic_df, home, paired)
                                       .drop(columns=['paired_airport'])
                                       .rename(columns={"pax_total": "Passengers"}), nixtla_model.predict(nb_days)])),
                     v1='Passengers', v2='RandomForestRegressor', secondary_y=False, covid_zone=True);   
    st.write(fig)

else :     
    fig = draw_ts_multiple(data, 'pax_total', covid_zone=True)
    st.write(fig)