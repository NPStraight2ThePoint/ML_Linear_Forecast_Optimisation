import pandas as pd
import numpy as np
import json
from datetime import datetime
from sqlalchemy import create_engine
from scipy.optimize import minimize
import psycopg2

conn = psycopg2.connect(
    dbname="Yahoo_Finance_API",
    user="postgres",
    password="Arxidolemios39",
    host="localhost",
    port=5432
)

# 1. SQL to get 10 unique settings
settings_sql = """
SELECT window_type, train_until, forecast_for, training_days_used, delta_days
FROM unique_settings_across_tickers
;
"""

settings_df = pd.read_sql(settings_sql, conn)

for idx, row in settings_df.iterrows():
    window_type = row['window_type']
    train_until = row['train_until']
    forecast_for = row['forecast_for']
    training_days_used = row['training_days_used']
    delta_days = row['delta_days']

    # Convert to datetime
    train_date = pd.to_datetime(train_until)
    forecast_date = pd.to_datetime(forecast_for)

    # 2. Query forecasts
    query = f"""
        SELECT *
        FROM simple_linear_forecasts
        WHERE window_type = '{window_type}'
          AND train_until = '{train_until}'
          AND forecast_for = '{forecast_for}'
          AND training_days_used = {training_days_used}
          AND delta_days = {delta_days};
    """
    forecast_df = pd.read_sql(query, conn)

    # 3. Calculate training window start date
    training_days = forecast_df['training_days_used'].iloc[0]
    training_window_start = (train_date - pd.Timedelta(days=training_days)).strftime('%Y-%m-%d')
    training_window_end = train_date.strftime('%Y-%m-%d')

    # 4. Get tickers and query prices for training window
    tickers = forecast_df['ticker'].unique().tolist()
    placeholders = ', '.join(['%s'] * len(tickers))

    price_training_query = f"""
    SELECT ticker, date, close
    FROM asx_etf_ohlcv
    WHERE date BETWEEN %s AND %s
    AND ticker IN ({placeholders})
    ORDER BY ticker, date
    """
    price_training_df = pd.read_sql(price_training_query, conn, params=[training_window_start, training_window_end] + tickers)

    price_training_df['date'] = pd.to_datetime(price_training_df['date'])
    max_date_idx = price_training_df.groupby('ticker')['date'].idxmax()
    current_price_df = price_training_df.loc[max_date_idx, ['ticker', 'close']].rename(columns={'close': 'current_price'})

    # 5. Merge forecast with current price and calculate expected return
    merged_df = forecast_df.merge(current_price_df, on='ticker', how='left')
    merged_df['expected_return'] = (merged_df['forecast_price'] - merged_df['current_price']) / merged_df['current_price']
    result_df = merged_df[['ticker', 'current_price', 'forecast_price', 'expected_return', 'stdev_delta_return']]

    # 6. Calculate covariance matrix
    price_training_df = price_training_df.sort_values(['ticker', 'date'])
    price_training_df['daily_return'] = price_training_df.groupby('ticker')['close'].pct_change()
    returns_df = price_training_df.pivot(index='date', columns='ticker', values='daily_return').dropna(how='any')
    cov_matrix_daily = returns_df.cov()

    cov_matrix_scaled = cov_matrix_daily * delta_days

    # 7. Get risk-free rate and scale
    train_until_date = train_date.strftime('%Y-%m-%d')
    query_rfr = """
    SELECT risk_free_rate
    FROM risk_free_rates
    WHERE date = %s
    """
    rfr_df = pd.read_sql(query_rfr, conn, params=[train_until_date])
    risk_free_rate = rfr_df['risk_free_rate'].iloc[0]
    trading_days_per_year = 252
    risk_free_rate_delta = (1 + risk_free_rate) ** (delta_days / trading_days_per_year) - 1

    # 8. Define Sharpe ratio functions for optimization
    def sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate_delta):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return (portfolio_return - risk_free_rate_delta) / portfolio_vol

    expected_returns = result_df['expected_return'].values
    cov_matrix = cov_matrix_scaled.values
    n = len(expected_returns)

    def neg_sharpe_ratio(weights):
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate_delta) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.ones(n) / n

    result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
        print("Optimal weights (%):", np.round(optimal_weights * 100, 2))
        final_sharpe = -result.fun
        final_return = np.dot(optimal_weights, expected_returns)
        final_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        print(f"Expected portfolio return: {final_return:.6f}")
        print(f"Portfolio volatility: {final_vol:.6f}")
        print(f"Max Sharpe Ratio: {final_sharpe:.6f}")
    else:
        print("Optimization failed:", result.message)
        continue  # skip to next iteration if optimization fails

    # 9. Query realized prices for holding period
    realized_prices_query = f"""
    SELECT ticker, date, close
    FROM asx_etf_ohlcv
    WHERE date BETWEEN %s AND %s
    AND ticker IN ({placeholders})
    ORDER BY ticker, date
    """
    realized_prices_df = pd.read_sql(realized_prices_query, conn, params=[train_until_date, forecast_date.strftime('%Y-%m-%d')] + tickers)
    realized_prices_df['date'] = pd.to_datetime(realized_prices_df['date'])

    buy_prices_df = realized_prices_df[realized_prices_df['date'] == train_date][['ticker', 'close']].rename(columns={'close': 'buy_price'})
    sell_prices_df = realized_prices_df[realized_prices_df['date'] == forecast_date][['ticker', 'close']].rename(columns={'close': 'sell_price'})

    weights_df = pd.DataFrame({'ticker': result_df['ticker'], 'weight': optimal_weights})

    portfolio_df = weights_df.merge(buy_prices_df, on='ticker').merge(sell_prices_df, on='ticker')
    portfolio_df['realized_return'] = (portfolio_df['sell_price'] - portfolio_df['buy_price']) / portfolio_df['buy_price']

    portfolio_realized_return = (portfolio_df['weight'] * portfolio_df['realized_return']).sum()

    # Calculate portfolio realized volatility
    daily_returns_df = realized_prices_df.pivot(index='date', columns='ticker', values='close').pct_change().dropna()
    holding_period_returns = daily_returns_df.loc[train_until_date:forecast_date]
    weights_series = pd.Series(optimal_weights, index=weights_df['ticker'])
    portfolio_daily_returns = holding_period_returns.dot(weights_series)
    portfolio_realized_vol = portfolio_daily_returns.std() * np.sqrt(delta_days)

    realized_sharpe_ratio = (portfolio_realized_return - risk_free_rate_delta) / portfolio_realized_vol

    print(f"Portfolio realized return over {delta_days} days: {portfolio_realized_return:.6f}")
    print(f"Portfolio realized volatility over {delta_days} days: {portfolio_realized_vol:.6f}")
    print(f"Portfolio realized Sharpe Ratio: {realized_sharpe_ratio:.6f}")

    # 10. Prepare insight row and insert into DB
    ticker_weights_dict = dict(zip(weights_df['ticker'], weights_df['weight']))

    row = {
        'window_type': window_type,
        'trained_from_date': training_window_start,
        'train_until_date': train_date.strftime('%Y-%m-%d'),
        'forecast_date': forecast_date.strftime('%Y-%m-%d'),
        'training_days_used': training_days,
        'delta_days': delta_days,
        'ticker_weights': json.dumps(ticker_weights_dict),
        'expected_portfolio_return': final_return,
        'expected_portfolio_volatility': final_vol,
        'expected_max_sharpe_ratio': final_sharpe,
        'realized_return': portfolio_realized_return,
        'realized_volatility': portfolio_realized_vol,
        'realized_sharpe_ratio': realized_sharpe_ratio,
        'risk_free_rate': risk_free_rate,
        'risk_free_rate_scaled': risk_free_rate_delta,
        'run_date': datetime.now()
    }

    insight_df = pd.DataFrame([row])

    # SQLAlchemy engine setup - update your credentials here
    username = "postgres"
    password = "Arxidolemios39"
    host = "localhost"
    port = "5432"
    database = "Yahoo_Finance_API"
    engine = create_engine(f"postgresql://{username}:{password}@{host}:{port}/{database}")

    try:
        insight_df.to_sql(
            name="sharpe_ratio_insights",
            con=engine,
            if_exists="append",
            index=False,
            method="multi"
        )
        print("✅ Data inserted into 'sharpe_ratio_insights' successfully.")
    except Exception as e:
        print(f"❌ Failed to insert data into database: {e}")