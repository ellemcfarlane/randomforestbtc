import pandas as pd
from functools import reduce


def display(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            print(df)


if __name__ == '__main__':
    avg_block_size = pd.read_csv('features/avg-block-size.csv')
    blocks_size = pd.read_csv('features/blocks-size.csv')
    cost_per_txn = pd.read_csv('features/cost-per-transaction.csv')
    difficulty = pd.read_csv('features/difficulty.csv')
    txn_vol = pd.read_csv('features/estimated-transaction-volume.csv')
    hash_rate = pd.read_csv('features/hash-rate-2.csv')
    market_cap = pd.read_csv('features/market-cap-2.csv')
    market_price = pd.read_csv('features/market-price.csv')
    confirm_time = pd.read_csv('features/median-confirmation-time-2.csv')
    miners_revenue = pd.read_csv('features/miners-revenue.csv')
    n_transaction = pd.read_csv('features/n-transactions.csv')
    n_transaction_exclude_long = pd.read_csv('features/n-transactions-excluding-longer.csv')
    n_transaction_exclude_popular = pd.read_csv('features/n-transactions-excluding-popular.csv')
    txn_per_block = pd.read_csv('features/n-transactions-per-block-3.csv')
    unique_addresses = pd.read_csv('features/n-unique-addresses.csv')
    output_vol = pd.read_csv('features/output-volume.csv')
    total_bitcoins = pd.read_csv('features/total-bitcoins-2.csv')
    trade_volume = pd.read_csv('features/trade-volume.csv')
    txn_fees = pd.read_csv('features/transaction-fees-2.csv')

    df_names = [
        'market_price', 'avg_block_size', 'blocks_size', 'cost_per_txn', 'difficulty', 'txn_vol', 'hash_rate',
        'market_cap', 'confirm_time', 'miners_revenue', 'n_transaction','n_transaction_exclude_popular',
        'txn_per_block', 'output_vol', 'total_bitcoins', 'trade_volume', 'txn_fees'
    ]

    dataframes = [
        market_price, avg_block_size, blocks_size, cost_per_txn, difficulty, txn_vol, hash_rate,
        market_cap, confirm_time, miners_revenue, n_transaction, n_transaction_exclude_popular,
        txn_per_block, output_vol, total_bitcoins, trade_volume, txn_fees
    ]

    for i in range(len(df_names)):
        dataframes[i].columns = ['date', df_names[i]]

    df_final = reduce(lambda left, right: pd.merge(left, right, on='date'), dataframes)
    df_final.to_csv('bitcoin_final.csv')
