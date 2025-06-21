# Efficient Frontier & Stock Portfolio Optimizer using Python (sci.py) with Monte Carlo Simulation based Stress-Testing

--> This is a **quantitative finance project** that implements **Modern Portfolio Theory (MPT)** using Python.

It constructs an **efficient frontier** based on user-selected Indian stocks (listed on NSE), and generates two optimal portfolios:

- Maximum Sharpe Ratio Portfolio (Risk-Adjusted Return Optimal)
- Minimum Variance Portfolio (Low Risk Optimal)

It then **stress-tests** these portfolios using **Monte Carlo Simulation** based on **Geometric Brownian Motion (GBM)** to calculate:

- 🔻 **VaR** (Value at Risk)
- 🔺 **CVaR** (Conditional Value at Risk)

# Key Concepts Used
- Modern Portfolio Theory (Markowitz)
- Efficient Frontier Visualization
- Mean-Variance Optimization (via Scipy)
- Monte Carlo Simulations for 1-Year Horizon
- Gaussian Assumption of Returns (for simplicity)

# How It Works
1. Edit the `tickers` list in the notebook. For Indian NSE stocks, use the `.NS` suffix (e.g., `INFY.NS`, `RELIANCE.NS`).
2. Run the notebook in [Jupyter Notebook](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/).
3. See the efficient frontier and optimal weights for two portfolios.
4. Monte Carlo simulation will estimate final values and risk metrics (VaR and CVaR).

# Example

The example output PNGs are attached.

# Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

# File Structure
```
final.ipynb         # Main notebook with full logic
README.md           # Project documentation
requirements.txt    # Python libraries required
```

# License
This project is open for educational use. Please credit the author if reusing.

Enjoy analyzing Indian markets with quant power!
