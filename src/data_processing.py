import pandas as pd
import numpy as np

def get_lookup_maps_and_stats(train_df):
    """
    Computes spatial lookup centroids and global feature statistics from training data.
    """
    clean_train = train_df.copy()
    
    # Replace empty/whitespace-only strings with NaN
    clean_train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    # Coerce numeric types and handle corrupted negative weight values
    if 'weight' in clean_train.columns:
        clean_train['weight'] = pd.to_numeric(clean_train['weight'], errors='coerce').abs()
    if 'market_index' in clean_train.columns:
        clean_train['market_index'] = pd.to_numeric(clean_train['market_index'], errors='coerce')
    
    # Compute mean geographic coordinates per pickup and delivery location
    pickup_map = clean_train.groupby('pickup')[['pickup_lat', 'pickup_lon']].mean().to_dict('index')
    delivery_map = clean_train.groupby('delivery')[['delivery_lat', 'delivery_lon']].mean().to_dict('index')
    
    # Compute fallback global baselines
    global_stats = {
        'pickup_lat': clean_train['pickup_lat'].mean(),
        'pickup_lon': clean_train['pickup_lon'].mean(),
        'delivery_lat': clean_train['delivery_lat'].mean(),
        'delivery_lon': clean_train['delivery_lon'].mean(),
        'weight': clean_train['weight'].median(),
        'market_index': clean_train['market_index'].mean(),
        'quote_signal': pd.to_numeric(clean_train['quote_signal'], errors='coerce').mean() if 'quote_signal' in clean_train.columns else 0.0
    }
    return pickup_map, delivery_map, global_stats

def handle_missing_and_impute(df, pickup_map, delivery_map, global_stats):
    """
    Cleans whitespaces, coerces numeric columns, and imputes missing features.
    """
    df = df.copy()
    
    # Convert empty whitespace strings to NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    # Safe numeric coercion and imputation
    if 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').abs()
        df['weight'] = df['weight'].fillna(global_stats['weight'])

    if 'market_index' in df.columns:
        df['market_index'] = pd.to_numeric(df['market_index'], errors='coerce')
        df['market_index'] = df['market_index'].fillna(global_stats['market_index'])

    if 'quote_signal' in df.columns:
        df['quote_signal'] = pd.to_numeric(df['quote_signal'], errors='coerce')
        df['quote_signal'] = df['quote_signal'].fillna(global_stats['quote_signal'])

    # Impute missing coordinates using dictionary mapping
    if 'pickup_lat' not in df.columns:
        pickup_lat_dict = {city: coords['pickup_lat'] for city, coords in pickup_map.items()}
        pickup_lon_dict = {city: coords['pickup_lon'] for city, coords in pickup_map.items()}
        delivery_lat_dict = {city: coords['delivery_lat'] for city, coords in delivery_map.items()}
        delivery_lon_dict = {city: coords['delivery_lon'] for city, coords in delivery_map.items()}

        df['pickup_lat'] = df['pickup'].map(pickup_lat_dict).fillna(global_stats['pickup_lat'])
        df['pickup_lon'] = df['pickup'].map(pickup_lon_dict).fillna(global_stats['pickup_lon'])
        df['delivery_lat'] = df['delivery'].map(delivery_lat_dict).fillna(global_stats['delivery_lat'])
        df['delivery_lon'] = df['delivery'].map(delivery_lon_dict).fillna(global_stats['delivery_lon'])

    if 'market_index' not in df.columns:
        df['market_index'] = global_stats['market_index']
    if 'quote_signal' not in df.columns:
        df['quote_signal'] = global_stats['quote_signal']

    return df