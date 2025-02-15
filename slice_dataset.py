import pandas as pd

df = pd.read_csv('./diffusiondb.csv')
data = df['prompt'].astype(str)

total_rows = len(data)
split1 = int(0.6 * total_rows)
split2 = int(0.8 * total_rows)

data_0_60 = data.iloc[:split1]
data_60_80 = data.iloc[split1:split2]
data_80_100 = data.iloc[split2:]

data_0_60.to_csv('./data/diffusiondb_0_60.csv', index=False, header=True)
data_60_80.to_csv('./data/diffusiondb_60_80.csv', index=False, header=True)
data_80_100.to_csv('./data/diffusiondb_80_100.csv', index=False, header=True)