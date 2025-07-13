import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from utils import ORIGINAL_PRICES

def generate_forecast_universe(
    csv_path: str,
    output_path: str,
    min_training_days: int = 252,
    forecast_step: int = 5,  # 1 week = 5 trading days
):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["adjclose"]).reset_index(drop=True)

    prices = df["adjclose"].values
    dates = df["date"].values
    ticker = df["ticker"].iloc[0] if "ticker" in df.columns else os.path.basename(csv_path).split(".")[0]
    total_days = len(df)

    results = []

    # Step 1: slide the training start point (inception_offset)
    for start_idx in range(0, total_days - min_training_days - forecast_step + 1, forecast_step):
        train_end_idx = start_idx + min_training_days

        # Step 2: expanding training window
        while train_end_idx + forecast_step <= total_days:
            x_train = np.arange(start_idx, train_end_idx).reshape(-1, 1)
            y_train = prices[start_idx:train_end_idx]

            model = LinearRegression()
            model.fit(x_train, y_train)

            # Training diagnostics
            r_squared = model.score(x_train, y_train)
            slope = model.coef_[0]
            intercept = model.intercept_

            # Step 3: test forecast horizons from 1w, 2w, ... up to available data
            max_horizon_steps = (total_days - train_end_idx) // forecast_step
            for h in range(1, max_horizon_steps + 1):
                horizon_days = h * forecast_step
                forecast_idx = train_end_idx + horizon_days - 1

                if forecast_idx >= total_days:
                    continue

                x_pred = np.array([[train_end_idx + horizon_days - 1]])
                predicted = model.predict(x_pred)[0]
                actual = prices[forecast_idx]

                error = predicted - actual
                abs_pct_error = abs(error / actual)
                mae = abs(error)
                mse = error ** 2
                mape = abs_pct_error  # same as abs % error

                results.append({
                    "ticker": ticker,
                    "train_start_date": dates[start_idx],
                    "train_end_date": dates[train_end_idx - 1],
                    "training_days": train_end_idx - start_idx,
                    "horizon_days": horizon_days,
                    "forecast_date": dates[forecast_idx],
                    "predicted_price": predicted,
                    "actual_price": actual,
                    "error": error,
                    "abs_pct_error": abs_pct_error,
                    "mae": mae,
                    "mse": mse,
                    "mape": mape,
                    "r_squared": r_squared,
                    "slope": slope,
                    "intercept": intercept
                })

            train_end_idx += forecast_step  # expand training window

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Forecasts saved to: {output_path}")


# === USAGE ===
ticker = '1GOV.AX'
input_csv = ORIGINAL_PRICES / f'{ticker}.csv'
output_csv = ORIGINAL_PRICES / f'{ticker}_forecasts.csv'

generate_forecast_universe(str(input_csv), str(output_csv))
