import yfinance as yf

data = yf.download(['MSFT', 'AAPL', 'GOOG'], period='1mo')
print(data.tail())