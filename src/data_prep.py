import numpy as np
import pandas as pd
import os
import pickle
from config import data_path, result_path, dates
from plot_curves import plot_forward_curve, plot_raw_smile


if __name__ == "__main__":
    data_packs = {date: {} for date in dates}

    for date in dates:
        xls = pd.ExcelFile(f"{data_path}/aapl_{date}.xlsx")
        # spot price
        spot = pd.read_excel(xls, sheet_name="spot", index_col=0).values[0, 0]
        # forward prices
        if "forward" not in xls.sheet_names:
            # the special case for "aapl_2025-04-08.xlsx"
            df_fwd  = pd.read_excel(xls, sheet_name="fwd").T.iloc[1:]
        else:
            df_fwd  = pd.read_excel(xls, sheet_name="forward").T.iloc[1:]
        df_fwd.columns = ["forward"]
        df_fwd.index = pd.to_datetime(df_fwd.index)
        df_fwd = df_fwd.sort_index()
        # implied vol grid
        df_vol = pd.read_excel(xls, sheet_name="gridvol", index_col=0).rename_axis(None)
        df_vol.index = pd.to_datetime(df_vol.index)
        df_vol = df_vol.sort_index()
        # expiry dates
        expiries = np.intersect1d(df_vol.index, df_fwd.index)

        # plot the forward curve
        fwd_path = os.path.join(result_path, 'forward_curve')
        if not os.path.exists(fwd_path):
            os.makedirs(fwd_path)
        plot_forward_curve(fwd_path, date, spot, df_fwd)
        # plot the implied volatility against the strikes
        smile_path = os.path.join(result_path, 'raw_smile')
        if not os.path.exists(smile_path):
            os.makedirs(smile_path)
        
        for expiry in expiries:
            expiry_str = pd.to_datetime(expiry).strftime('%Y-%m-%d')
            row = df_vol.loc[expiry]
            # Extract strikes and implied vols from that row, ignoring NaNs
            strikes = row.dropna().index.to_list()
            ivs = row.dropna()
            smile_path_date = os.path.join(smile_path, date)
            if not os.path.exists(smile_path_date):
                os.makedirs(smile_path_date)
            plot_raw_smile(smile_path_date, date, expiry_str, strikes, ivs.values)
            # forward price for this expiry F
            F = df_fwd.loc[expiry].values[0]
            # time to expiry T
            T = (expiry - pd.Timestamp(date)).days / 365.0
            # log-forward moneyness k
            k = np.log(strikes / F)
            # total variance w = iv^2 * T
            w = ivs**2 * T
            # store the data
            data_packs[date][expiry_str] = {
                "spot": spot,
                "F": F,
                "T": T,
                "strikes": strikes,
                "k": k,
                "w": w
            }

    # Save the data packs to data_path, as a usable format, can be read as a dict, save as a whole
    with open(f"{data_path}/data_packs.pkl", "wb") as f:
        pickle.dump(data_packs, f)

    # # How to load the data packs
    # with open(f"{data_path}/data_packs.pkl", "rb") as f:
    #     data_packs = pickle.load(f)
