import pandas as pd
from sqlalchemy import create_engine, MetaData
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import os

# --- Settings ---
folder_path = 'temp_data/linear_forecasts_simple'  # your folder with CSVs
table_name = 'simple_linear_forecasts'
db_url = 'postgresql://postgres:Arxidolemios39@localhost:5432/Yahoo_Finance_API'  # replace this

engine = create_engine(db_url)
metadata = MetaData()
metadata.reflect(bind=engine)
forecast_table = metadata.tables[table_name]

for filename in os.listdir(folder_path):
    if not filename.endswith('.csv'):
        continue

    file_path = os.path.join(folder_path, filename)
    print(f'📂 Processing: {filename}')

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()

        # Clean date columns (strip whitespace)
        df['train_until'] = df['train_until'].astype(str).str.strip()
        df['forecast_for'] = df['forecast_for'].astype(str).str.strip()

        # Try parsing dates with format if consistent, else fallback
        try:
            df['train_until'] = pd.to_datetime(df['train_until'], format='%Y-%m-%d', errors='raise')
            df['forecast_for'] = pd.to_datetime(df['forecast_for'], format='%Y-%m-%d', errors='raise')
        except Exception:
            print(f"⚠️ Warning: inconsistent date format in {filename}, trying generic parse")
            df['train_until'] = pd.to_datetime(df['train_until'], errors='coerce')
            df['forecast_for'] = pd.to_datetime(df['forecast_for'], errors='coerce')

        # Check for NaT after parsing
        if df['train_until'].isna().any() or df['forecast_for'].isna().any():
            print(f"⚠️ Warning: NaT detected in date parsing for {filename}")

        df['training_days_used'] = df['training_days_used'].astype(int)
        df['delta_days'] = df['delta_days'].astype(int)

        # Reorder columns if needed
        expected_columns = [
            'ticker', 'window_type', 'train_until', 'forecast_for',
            'training_days_used', 'delta_days', 'forecast_price', 'actual_price',
            'error', 'r_squared', 'slope', 'intercept', 'mae', 'mse', 'mape', 'stdev_delta_return'
        ]
        df = df[expected_columns]

        # Insert with conflict handling (update if exists)
        records = df.to_dict(orient='records')
        stmt = insert(forecast_table).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'window_type', 'train_until', 'forecast_for', 'delta_days'],
            set_={col: stmt.excluded[col] for col in df.columns if col not in ['ticker', 'window_type', 'train_until', 'forecast_for', 'delta_days']}
        )

        with engine.begin() as conn:
            conn.execute(stmt)

        print(f'✅ Inserted: {filename} ({len(df)} rows)')

    except Exception as e:
        print(f'❌ Error in {filename}: {e}')