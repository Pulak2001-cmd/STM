import pandas_datareader as pdr
key = "2b538ef8ec0bc80e04375a71ceda7083f781a14a"
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL1.csv')