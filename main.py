import os
import subprocess
import pandas as pd
import numpy as np

from src.data_processing import get_lookup_maps_and_stats, handle_missing_and_impute
from src.feature_engineering import add_features
from src.model import train_and_predict

def main():
    print("Starting Freight Rate Prediction Pipeline...")

    dec_path = 'data/december-chart-inputs.csv' if os.path.exists('data/december-chart-inputs.csv') else 'data/december_chart_inputs.csv'

    # 1. Load raw datasets
    train_df = pd.read_csv('data/train-test.csv')
    val_df = pd.read_csv('data/validation.csv')
    dec_df = pd.read_csv(dec_path)

    # 2. Filter extreme target rate outliers from training set
    train_df['posted_rate'] = pd.to_numeric(train_df['posted_rate'], errors='coerce')
    train_df['distance'] = pd.to_numeric(train_df['distance'], errors='coerce')
    train_df['rate_per_mile'] = train_df['posted_rate'] / np.maximum(train_df['distance'], 1.0)
    train_df = train_df[(train_df['posted_rate'] <= 18000) & (train_df['rate_per_mile'] <= 15.0)].reset_index(drop=True)

    # 3. Clean whitespaces, negative weights, and impute missing values
    pickup_map, delivery_map, global_stats = get_lookup_maps_and_stats(train_df)
    train_df = handle_missing_and_impute(train_df, pickup_map, delivery_map, global_stats)
    val_df = handle_missing_and_impute(val_df, pickup_map, delivery_map, global_stats)
    dec_df = handle_missing_and_impute(dec_df, pickup_map, delivery_map, global_stats)

    # 4. Feature engineering
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    dec_df = add_features(dec_df)

    # 5. Define numerical and categorical feature sets
    num_features = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'log_distance', 'dist_sq', 'calc_distance_km', 'weight',
        'year', 'quarter', 'month', 'day', 'dayofweek', 'dayofyear', 'is_weekend',
        'market_index', 'quote_signal',
        'dist_x_market', 'dist_x_quote', 'sin_day', 'cos_day'
    ]
    cat_features = ['pickup', 'delivery', 'equipment', 'route']

    # 6. Train model and generate predictions
    val_preds, dec_preds = train_and_predict(train_df, val_df, dec_df, num_features, cat_features)

    # 7. Format and export validation predictions
    val_output = pd.DataFrame({
        'load_id': val_df['load_id'],
        'predicted_rate': np.round(val_preds, 2)
    })
    val_output.to_csv('validation_predictions.csv', index=False)

    dec_original = pd.read_csv(dec_path)
    dec_original['predicted_rate'] = np.round(dec_preds, 2)
    
    out_dec_path = 'data/december_chart_inputs.csv'
    dec_original[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']].to_csv(out_dec_path, index=False)

    print("\n Successfully saved output prediction files.")

    # 8. Execute evaluation script score.py
    print("Running evaluation score.py script...")
    subprocess.run([
        "python", "score.py",
        "--predictions", "validation_predictions.csv",
        "--december-predictions", out_dec_path
    ])

if __name__ == "__main__":
    main()