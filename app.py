
import streamlit as st
from utils.data_utils import retrieve_data
from utils.model import train_model, mae
from utils.geo_id import GEO_IDS
from utils.forecast import predict_df
from matplotlib import pyplot as plt
from pandas import to_datetime

st.title("Prophylax: Covid19 new cases forecaster")
st.markdown("Select a country to view the new cases forcasted by Prophylax,\
        You can also select the regression algorithm and the window length (click/tap the arrow if you can't see the settings).\
        The MAE metric helps us estimate the accuracy of the forecast (lower is better). \
        Try adjusting the settings to lower the MAE value and get better results.\
        The Github repository of the app is available [here](https://github.com/ShahNewazKhan/prophylax).")

country = st.sidebar.selectbox(
    label = "Select a Region", 
    index = 77, 
    options = GEO_IDS)

regressor = st.sidebar.selectbox(
    "Select a Regression Algorithm",   
    [
        'Linear Regression',
        'Random Forest', 
        'Gradient Boosting', 
        'XGBoost',
        'Support Vector Machines', 
        'Extra Trees' 
    ]
)                    

window_length = st.sidebar.slider(label = 'Look back days',
                                  min_value = 1, value = 30)


X_train, X_test, y_train, y_test = retrieve_data(window_length, [country])

st.subheader(f'Forecasting with {regressor}')


model = train_model(regressor, X_train, y_train)
pdf, hist_df = predict_df(model, regions=[country], start_date_str='2020-11-16', end_date_str='2020-12-06', lookback_days=window_length)

pdf.index = to_datetime(pdf['Date'])
hist_df.index = to_datetime(hist_df['Date'])

st.text(f"MAE: {mae(pdf['PredictedDailyNewCases'],hist_df['NewCases'])}")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(pdf['PredictedDailyNewCases'], label='Predictions')
ax.plot(hist_df['NewCases'], label='Actual')
ax.legend()
st.pyplot(fig)

st.text('Predictions Dataframe')
st.dataframe(pdf, width=800, height=200)

st.text('Actuals Dataframe')
st.dataframe(hist_df, width=800, height=200)
