import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def recalc_stdev_delta_return(row):
    ticker = row['ticker']
    train_until = pd.to_datetime(row['train_until'])
    training_days_used = int(row['training_days_used'])
    delta_days = int(row['delta_days'])
    # Filter closes_df for the ticker and dates <= train_until
    df_ticker = closes_df[(closes_df['ticker'] == ticker) & (pd.to_datetime(closes_df['date']) <= train_until)].copy()
    df_ticker = df_ticker.sort_values('date').reset_index(drop=True)
    if len(df_ticker) < training_days_used:
        # Not enough data, return NaN or keep original value
        return np.nan
    train_data = df_ticker.tail(training_days_used)
    # Calculate daily returns
    train_data['return'] = train_data['close'].pct_change()
    realized_stdev = train_data['return'].std()
    if pd.isna(realized_stdev) or realized_stdev == 0:
        # fallback: use total return scaled
        total_return = train_data['close'].iloc[-1] / train_data['close'].iloc[0] - 1
        realized_stdev = abs(total_return) / np.sqrt(training_days_used)
    # Scale stdev by sqrt(delta_days)
    scaled_stdev = realized_stdev * np.sqrt(delta_days)
    return scaled_stdev

# --- Configuration ---
db_url = 'postgresql://postgres:Arxidolemios39@localhost:5432/Yahoo_Finance_API'  # UPDATE THIS
closes_table = 'asx_etf_ohlcv'
forecasts_table = 'simple_linear_forecasts'
output_path = 'temp_data/linear_forecasts_simple'
os.makedirs(output_path, exist_ok=True)

# Forecast settings
expanding_horizons = [5, 10, 21, 63, 126, 252]
window_to_horizons = {
    21: [5, 10],
    63: [5, 10, 21],
    126: [5, 10, 21, 63],
    252: [5, 10, 21, 63, 126],
    504: [5, 10, 21, 63, 126, 252],
}
min_training_days = 252
save_interval = 100

# SQL extraction
engine = create_engine(db_url)
closes_df = pd.read_sql(f'SELECT * FROM {closes_table}', engine)

try:
    forecasts_df = pd.read_sql(f'SELECT * FROM {forecasts_table}', engine)
except Exception:
    forecasts_df = pd.DataFrame()

tickers = closes_df['ticker'].unique()

for ticker in tickers:
    df = closes_df[closes_df['ticker'] == ticker].copy()
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)

    existing = forecasts_df[forecasts_df['ticker'] == ticker] if not forecasts_df.empty else pd.DataFrame()

    latest_train_until = {}
    for win in ['expanding'] + [f'sliding_{w}' for w in window_to_horizons]:
        if not existing.empty:
            max_date = existing[existing['window_type'] == win]['train_until'].max()
            latest_train_until[win] = pd.to_datetime(max_date) if pd.notnull(max_date) else None
        else:
            latest_train_until[win] = None

    all_results = []
    df['date'] = pd.to_datetime(df['date'])

    # ========== 1. EXPANDING WINDOW ==========
    for i in range(min_training_days, len(df) - max(expanding_horizons)):
        forecast_date = df.iloc[i]['date']
        if latest_train_until['expanding'] and forecast_date <= latest_train_until['expanding']:
            continue

        train = df.iloc[:i].copy()
        training_days_used = len(train)
        train['days'] = (train['date'] - train['date'].min()).dt.days

        X_train = train[['days']]
        y_train = train['close']

        model = LinearRegression()
        model.fit(X_train, y_train)

        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(X_train, y_train)

        for h in expanding_horizons:
            future_index = i + h
            if future_index >= len(df):
                continue

            forecast_base_day = (df.iloc[i]['date'] - train['date'].min()).days
            forecast_input_df = pd.DataFrame([[forecast_base_day + h]], columns=['days'])
            forecast_price = model.predict(forecast_input_df)[0]

            target_date = df.iloc[future_index]['date']
            actual_price = df.iloc[future_index]['close']

            if pd.isna(actual_price):
                continue  # Skip rows with missing actual prices

            error = forecast_price - actual_price
            mse = mean_squared_error([actual_price], [forecast_price])
            mae = mean_absolute_error([actual_price], [forecast_price])
            mape = abs(error) / actual_price * 100 if actual_price != 0 else None

            try:
                train_with_future = train.copy()
                train_with_future['future_return'] = train_with_future['close'].shift(-h) / train_with_future[
                    'close'] - 1
                valid_returns = train_with_future['future_return'].dropna()

                delta_stdev = valid_returns.std()

                # Always fallback if stdev is zero or NaN
                if pd.isna(delta_stdev) or delta_stdev == 0:
                    total_return = train['close'].iloc[-1] / train['close'].iloc[0] - 1
                    train_days = len(train)
                    delta_stdev = abs(total_return) / np.sqrt(train_days) * np.sqrt(h)

            except Exception as e:
                print(f"Exception caught: {e}")
                total_return = train['close'].iloc[-1] / train['close'].iloc[0] - 1
                train_days = len(train)
                delta_stdev = abs(total_return) / np.sqrt(train_days) * np.sqrt(h)


            except Exception as e:
                print(f"Exception caught: {e}")
                total_return = train['close'].iloc[-1] / train['close'].iloc[0] - 1
                train_days = len(train)
                delta_stdev = abs(total_return) / np.sqrt(train_days) * np.sqrt(h)

            all_results.append({
                'ticker': ticker,
                'window_type': 'expanding',
                'train_until': forecast_date,
                'forecast_for': target_date,
                'training_days_used': training_days_used,
                'delta_days': h,
                'forecast_price': round(forecast_price, 4),
                'actual_price': round(actual_price, 4),
                'error': round(error, 4),
                'r_squared': round(r_squared, 4),
                'slope': round(slope, 6),
                'intercept': round(intercept, 4),
                'mae': round(mae, 4),
                'mse': round(mse, 4),
                'mape': round(mape, 4) if mape is not None else None,
                'stdev_delta_return': round(delta_stdev, 6),
            })

    # ========== 2. SLIDING WINDOWS ==========
    for window_size, horizons in window_to_horizons.items():
        label = f'sliding_{window_size}'
        for i in range(0, len(df) - window_size - max(horizons)):

            forecast_date = df.iloc[i + window_size]['date']
            if latest_train_until[label] and forecast_date <= latest_train_until[label]:
                continue

            train = df.iloc[i:i + window_size].copy()
            training_days_used = window_size
            train['days'] = (train['date'] - train['date'].min()).dt.days
            X_train = train[['days']]
            y_train = train['close']

            model = LinearRegression()
            model.fit(X_train, y_train)

            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X_train, y_train)

            for h in horizons:
                forecast_index = i + window_size
                future_index = forecast_index + h
                if future_index >= len(df):
                    continue

                forecast_base_day = (df.iloc[forecast_index]['date'] - train['date'].min()).days
                forecast_input_df = pd.DataFrame([[forecast_base_day + h]], columns=['days'])
                forecast_price = model.predict(forecast_input_df)[0]

                target_date = df.iloc[future_index]['date']
                actual_price = df.iloc[future_index]['close']

                if pd.isna(actual_price):
                    continue  # Skip rows with missing actual prices

                error = forecast_price - actual_price
                mse = mean_squared_error([actual_price], [forecast_price])
                mae = mean_absolute_error([actual_price], [forecast_price])
                mape = abs(error) / actual_price * 100 if actual_price != 0 else None

                try:
                    train_with_future = train.copy()
                    train_with_future['future_return'] = train_with_future['close'].shift(-h) / train_with_future[
                        'close'] - 1
                    valid_returns = train_with_future['future_return'].dropna()

                    delta_stdev = valid_returns.std()

                    # Always fallback if stdev is zero or NaN
                    if pd.isna(delta_stdev) or delta_stdev == 0:
                        total_return = train['close'].iloc[-1] / train['close'].iloc[0] - 1
                        train_days = len(train)
                        delta_stdev = abs(total_return) / np.sqrt(train_days) * np.sqrt(h)

                except Exception as e:
                    print(f"Exception caught: {e}")
                    total_return = train['close'].iloc[-1] / train['close'].iloc[0] - 1
                    train_days = len(train)
                    delta_stdev = abs(total_return) / np.sqrt(train_days) * np.sqrt(h)

                all_results.append({
                    'ticker': ticker,
                    'window_type': label,
                    'train_until': forecast_date,
                    'forecast_for': target_date,
                    'training_days_used': training_days_used,
                    'delta_days': h,
                    'forecast_price': round(forecast_price, 4),
                    'actual_price': round(actual_price, 4),
                    'error': round(error, 4),
                    'r_squared': round(r_squared, 4),
                    'slope': round(slope, 6),
                    'intercept': round(intercept, 4),
                    'mae': round(mae, 4),
                    'mse': round(mse, 4),
                    'mape': round(mape, 4) if mape is not None else None,
                    'stdev_delta_return': round(delta_stdev, 6),
                })

    if all_results:
        df_results = pd.DataFrame(all_results)
        csv_path = os.path.join(output_path, f'{ticker}_forecast.csv')

        # Save initial forecast CSV
        df_results.to_csv(csv_path, index=False)
        print(f"✓ Saved forecast for {ticker} with {len(df_results)} new rows")

        # Now fix stdev_delta_return if needed, immediately for this ticker
        mask_fix = (df_results['stdev_delta_return'].isna()) | (df_results['stdev_delta_return'] == 0)
        if mask_fix.any():
            print(f"Post-processing: fixing {mask_fix.sum()} rows with NaN or 0 stdev_delta_return for {ticker}")
            for idx in df_results.loc[mask_fix].index:
                try:
                    new_stdev = recalc_stdev_delta_return(df_results.loc[idx])
                    if pd.notna(new_stdev):
                        df_results.at[idx, 'stdev_delta_return'] = round(new_stdev, 6)
                except Exception as e:
                    print(f"Error recalculating stdev_delta_return for index {idx} of {ticker}: {e}")

            # Save updated results again for this ticker
            df_results.to_csv(csv_path, index=False)
            print(f"✓ Updated stdev_delta_return saved for {ticker}")



