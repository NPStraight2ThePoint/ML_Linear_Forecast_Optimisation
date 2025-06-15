# ML_Linear_Forecast_Optimisation

# ASX ETF ML Pipeline

This repository contains the machine learning pipeline for ASX ETF data analysis and portfolio optimization. It is designed as an extension of the [ASX_ETF_Yahoo_Finance_ETL](https://github.com/yourusername/ASX_ETF_Yahoo_Finance_ETL) repository, which handles the data extraction and loading processes.

---

## Overview

The ML pipeline takes the cleaned historical OHLCV data extracted via the ETL workflow and applies advanced analytical methods to generate investment insights, including:

- Linear and non-linear price forecasting models
- Portfolio optimization based on Sharpe Ratio maximization
- Comparison of expected vs. realized portfolio metrics
- Storage of ML-generated insights back into the PostgreSQL database

This pipeline enables quantitative analysis and data-driven portfolio construction for ASX ETFs and can be adapted for other ticker universes.

---

## Key Features

- Scalable processing for large ticker universes
- Modular design to plug into existing ETL workflows
- Integration with PostgreSQL for seamless data management
- Optimization routines for portfolio weighting based on forecasted returns and risk

---

## Tech Stack

- Python (pandas, numpy, scipy, sklearn)
- PostgreSQL (database for historical data and insights)
- SQLAlchemy (database connection and ORM)
- Jupyter Notebooks / Scripts for experimentation and automation

---

## Usage

1. Ensure the ETL pipeline is run and data is loaded into the database.
2. Configure database connection settings in the ML scripts.
3. Run forecasting models and portfolio optimization scripts.
4. Review the results stored in the database or exported reports.

---

## Notes

- This repository is designed as a companion to the [ASX_ETF_Yahoo_Finance_ETL](https://github.com/yourusername/ASX_ETF_Yahoo_Finance_ETL) repo. 
- While tailored for ASX ETFs, the ML pipeline can be adapted for any ticker list with appropriate data.
- Future enhancements may include additional model types and risk management features.

---

## License

MIT License

---

Feel free to reach out or submit issues/PRs for improvements!
