# swc-simulation
SWC (Seabridge Gold Inc.) Bitcoin strategy simulation

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

- Oscillating cycles between 0.8 and decaying peaks over 1-3 month periods
- 3-zone distribution: 5% discount (0.8-1.3), 30% moderate (1.3-2.5), 65% high premium (2.5-decaying max)
- Decaying power law for peaks: max_mnav = 7.0 * (time_in_years + 1)^(-0.15)
- mNAV is dampened by dilution rate: for each 1% dilution, mNAV is reduced by 5%
- Multiple overlapping sine wave cycles (45-day, 180-day, 360-day periods)

## Stock Price

- Derived from mNAV = Market Cap / Bitcoin NAV
- Price = (mNAV * Bitcoin Price * Bitcoin Holdings) / Outstanding Shares

## Stock Implied Volatility

- Derived from Stock Price fluctuations, 30-day window

## Shares Outstanding 

- Derived from Initial Shares + Dilution

## Daily Trading Volume

- Volume correlated to mNAV cyclical model (higher mNAV = higher volume)
- Range: 1% to 5% of outstanding shares
- Low mNAV (≤1.0): 1.0%-2.0% volume
- Moderate mNAV (1.0-2.0): 2.0%-3.5% volume  
- High mNAV (>2.0): 3.5%-5.0% volume
- Aggregated across all SWC tickers: SWC.AQ, TSWCF, 3M8.F
- Small daily random noise (±5%)

## Daily Share Dilution

- Simplified dilution logic: 20% of daily trading volume when price increases from the previous day
- Dilution occurs only if mNAV (market cap to NAV ratio) is above 1.1
- Dilution only occurs on trading days (weekdays)
- Dilution rate tracked and used to dampen future mNAV (1% dilution = 5% mNAV reduction)
- Rate decays exponentially when no dilution occurs (5% decay per day)

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