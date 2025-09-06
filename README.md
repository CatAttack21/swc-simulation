# metaplanet-prediction
Metaplanet metric prediction

## Bitcoin Price

- Follow the Bitcoin Power law, price = 10**-17 * (days_since_genesis ** 5.8)
- Support at 0.5x, Resistance at 4x
- Superimpose 8 sinusoid functions over varying frequency/amplitude for price action

## Bitcoin CAGR

- Derived from BTC performance
- A Rolling window, year-over-year growth of Bitcoin Price
- Roughly swings between 10%-50% median value

## Bitcoin Holdings

- Derived from purchases
- ATM = BTC
- Bitcoin puts revenue (0.5% market cap annually, cash on hand to sell puts) = BTC
- Preferred Shares = BTC

## BTC per 1000 shares

- Derived from Bitcoin Holdings / Outstanding Shares

## mNAV

- Follows a power law, as described by Metaplanet presentations, theoretical_mcap = 35.1221 * (btc_value ** 0.89)
https://aqsmixvnbavrufgttwsn.supabase.co/storage/v1/object/public/media-resources/en/67fa9191-0efa-467e-8996-71a68ab3a62f/q1-2025-earnings-presentation-20250514T192554478Z.pdf

- 15% chance of any overshoot/undershoot (equal weight)
- Upside random overshoot between 33% and 400%
- Downside random overshoot between -50% and -20%
- Sinusoidal volatility based on random noise
- Mean reversion strength between 5% and 15%

## Stock Price

- Derived from mNAV = Market Cap / Bitcoin NAV
- Price = (mNAV * Bitcoin Price * Bitcoin Holdings) / Outstanding Shares

## Stock Implied Volatility

- Derived from Stock Price fluctuations, 30-day window

## Shares Outstanding 

- Derived from Initial Shares + Dilution

## Daily Trading Volume

- Takes the initial 3350 volume and normalizes the median to 10% of outstanding shares (Mimics MSTR performance through 2025)
- Weekly Volume factor between 20%-100%
- Apply a factor of mNAV to boost during high mNAV
- Add a component of random volatility noise, this dominates at low volume
- Bound final between 1% and 33% of outstanding shares

## Daily Share Dilution

- Dilution occurs only if the stock price increases from the previous day
- Dilution occurs only if mNAV (market cap to NAV ratio) is above 1.1
- Dilution is 20% of daily trading volume
- Dilution only occurs on trading days (weekdays)

## Preferred Shares

- S-curve adoption model of ATM that asymptotes to 100 Million shares
- $100 face value
- 5% annual yield

## Cumulative Preferred Share Dividends

- Payouts made quarterly, targeting 5% annual return
- Payouts made according to the shares outstanding
- Dividends are paid by common stock dilution, limits BTC purchases

## Market Cap to Cumulative Dividends Ratio

- A gross check on whether preferred shares and dividends are too high