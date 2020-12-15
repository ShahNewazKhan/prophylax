import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# Main source for the training data
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'

# For testing, restrict training data to that before a hypothetical predictor submission date
HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-11-15")

@st.cache
def retrieve_data(window_length: int, regions: list):
    """Retrieves the OxCGRT csv and loads it into a dataframe"""

    df = pd.read_csv(DATA_URL, 
                    parse_dates=['Date'],
                    encoding="ISO-8859-1",
                    dtype={"RegionName": str,
                            "RegionCode": str},
                    error_bad_lines=False)
    
    df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]
    
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
    
    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)    
    
    # Keep only columns of interest
    id_cols = ['CountryName',
            'RegionName',
            'GeoID',
            'Date']
    cases_col = ['NewCases']
    npi_cols = ['C1_School closing',
                'C2_Workplace closing',
                'C3_Cancel public events',
                'C4_Restrictions on gatherings',
                'C5_Close public transport',
                'C6_Stay at home requirements',
                'C7_Restrictions on internal movement',
                'C8_International travel controls',
                'H1_Public information campaigns',
                'H2_Testing policy',
                'H3_Contact tracing',
                'H6_Facial Coverings']
    df = df[id_cols + cases_col + npi_cols]
    
    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))
    
    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in npi_cols:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))
    
    # Set number of past days to use to make predictions
    nb_lookback_days = window_length

    # Create training data across all countries for predicting one day ahead
    X_cols = cases_col + npi_cols
    y_col = cases_col
    X_samples = []
    y_samples = []
    
    # if regions[0] == 'All':
    #     regions = df.GeoID.unique()
    
    for g in df.GeoID.unique():
        gdf = df[df.GeoID == g]
        all_case_data = np.array(gdf[cases_col])
        all_npi_data = np.array(gdf[npi_cols])

        # Create one sample for each day where we have enough data
        # Each sample consists of cases and npis for previous nb_lookback_days
        nb_total_days = len(gdf)
        for d in range(nb_lookback_days, nb_total_days - 1):
            X_cases = all_case_data[d-nb_lookback_days:d]

            # Take negative of npis to support positive
            # weight constraint in Lasso.
            X_npis = -all_npi_data[d - nb_lookback_days:d]

            # Flatten all input data so it fits Lasso input format.
            X_sample = np.concatenate([X_cases.flatten(),
                                    X_npis.flatten()])
            y_sample = all_case_data[d + 1]
            X_samples.append(X_sample)
            y_samples.append(y_sample)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples).flatten()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                        y_samples,
                                                        test_size=0.2,
                                                        random_state=301)

    return X_train, X_test, y_train, y_test
