import pandas as pd
import numpy as np

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Computes geodesic Haversine distance in kilometers between two lat/lon coordinates.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367.0 * c

def add_features(df):
    """
    Generates temporal calendar features, non-linear distance transformations, and interaction signals.
    """
    df = df.copy()
    
    # Parse date column safely and extract temporal/calendar features
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Calculate spatial distance and route interaction identifiers
    df['calc_distance_km'] = haversine_np(df['pickup_lon'], df['pickup_lat'], df['delivery_lon'], df['delivery_lat'])
    df['route'] = df['pickup'].astype(str) + '_' + df['delivery'].astype(str)

    # Non-linear transformations to capture non-linear distance curves
    df['log_distance'] = np.log1p(df['distance'])
    df['dist_sq'] = np.sqrt(df['distance'])
    
    # Feature interactions with market dynamics
    df['dist_x_market'] = df['distance'] * df['market_index']
    df['dist_x_quote'] = df['distance'] * df['quote_signal']
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

    return df