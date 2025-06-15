# ML_Linear_Forecast_Optimisation

This repository contains the machine learning pipeline for ASX ETF data analysis and portfolio optimization. It is designed as an extension of the [ASX_ETF_Yahoo_Finance_ETL](https://github.com/NPStraight2ThePoint/ASX_ETF_Yahoo_Finance_ETL) repository, which handles the data extraction and loading process.

---

## Key Features

- Scalable processing for large ticker universes
- Modular design to plug into existing ETL workflows
- Integration with PostgreSQL for seamless data management
- Optimization routines for portfolio weighting based on forecasted returns and risk

---

## Overview

The ML pipeline takes the cleaned historical OHLCV data extracted via the ETL workflow and applies advanced analytical methods to generate investment insights, including:

- OLS - Linear regression price targets for various settings (expanding/sliding training window, forward delta, in and out of sample duration
- Delta scaled standard deviation estimations.
- Model Parameters / Coefficients: Residual, Intercept, Slope
- Performance Metrics: R², MAE, MSE, MAPE
- optimal weights/max Sharpe Ratio in and out of sample -> Expected vs Actual universe mapping

---

## Tech Stack

- Python (pandas, numpy, scipy, sklearn)
- PostgreSQL (database)
- SQLAlchemy/psycopg2 (database connection and ORM)

---

## Usage

1. Ensure the ETL pipeline is run and data is loaded into the database.
2. Configure database connection settings in the ML scripts.
3. Run forecasting models and portfolio optimization scripts.
4. Review the results stored in the database or exported reports.

---

## Notes

- This repository is an extention of the [ASX_ETF_Yahoo_Finance_ETL](https://github.com/NPStraight2ThePoint/ASX_ETF_Yahoo_Finance_ETL) repo. 
- While tailored for ASX ETFs, the ML pipeline can be adapted for any ticker list with appropriate data.
- Future enhancements may include additional model types and risk management features.

---

### 🆔 Project Info

**Author:** *Nicholas Papadimitris*  
**Created on:** *15/06/2025 9:54 PM* (UTC)    
**Project ID:** `Finance_Project_NP_15_Jun2025`  
**GitHub:** [My GitHub](https://github.com/NPStraight2ThePoint)

📧 **Email:** nicholas.papadimitris@gmail.com  
💼 **LinkedIn:** [Nicholas Papadimitris](https://www.linkedin.com/in/nicholas-papadimitris/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

Feel free to reach out or submit issues/PRs for improvements!
