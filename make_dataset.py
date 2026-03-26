import pandas as pd
import os

print("Creating crypto dataset...")

folder = "archive"
all_data = []

coins = [
    "Bitcoin",
    "Ethereum",
    "Cardano",
    "Dogecoin",
    "Litecoin",
    "XRP",
    "Solana",
    "Polkadot",
    "ChainLink",
    "Stellar"
]

symbols = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Cardano": "ADA",
    "Dogecoin": "DOGE",
    "Litecoin": "LTC",
    "XRP": "XRP",
    "Solana": "SOL",
    "Polkadot": "DOT",
    "ChainLink": "LINK",
    "Stellar": "XLM"
}

for coin in coins:
    path = os.path.join(folder, f"coin_{coin}.csv")

    if not os.path.exists(path):
        print(f"⚠ {coin} not found — skipped")
        continue

    df = pd.read_csv(path)

    # pick useful columns
    df = df[['Date','Open','Marketcap','Volume']]

    # rename columns
    df.rename(columns={
        'Date':'date',
        'Open':'price',
        'Marketcap':'market_cap',
        'Volume':'volume'
    }, inplace=True)

    df['cryptocurrency'] = coin
    df['symbol'] = symbols[coin]

    # percent change
    df['percent_change_24h'] = df['price'].pct_change() * 100

    all_data.append(df)
    print(f"✔ {coin} processed")

# merge all coins
final_df = pd.concat(all_data, ignore_index=True)

# convert date
final_df['date'] = pd.to_datetime(final_df['date'])

# sort by date
final_df = final_df.sort_values(by='date')

# save
os.makedirs("dataset", exist_ok=True)
final_df.to_csv("dataset/crypto_data.csv", index=False)

print("\n🎉 SUCCESS!")
print("File created: dataset/crypto_data.csv")
print("Total rows:", final_df.shape[0])
print("Total columns:", final_df.shape[1])