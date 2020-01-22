import numpy as np
import pandas as pd
from functools import reduce


def clean_pcts(x):
    if type(x) in (int, float):
        return float(x)
    else:
        return float(x.replace('%', '').strip()) / 100    


def merge_seasons_data(dfs, on=('Season', 'playerid', 'Name')):
    right = '_drop1'
    merged = reduce(lambda x, y: pd.merge(x, y, on=on, suffixes=('', right)), dfs)
    merged.drop([col for col in merged.columns if right in col], axis=1, inplace=True)
    return merged 
    
    
def wOBA(row):
    bbwgt = 0.69
    hbpwgt = 0.719
    wgt1b = 0.870
    wgt2b = 1.217
    wgt3b = 1.529
    hrwgt = 1.940
    return bbwgt * row['BB'] + hbpwgt * row['HBP'] + wgt1b * row['1B'] + wgt2b * row['2B'] \
           + wgt3b * row['3B'] + hrwgt * row['HR'] / (row['AB'] + row['BB'] - row['IBB'] + row['SF'] + row['HBP'])


def wRAA(row):
    lgwOBA = .321
    wOBA_scale = 1.21
    return ((row['wOBA'] - lgwOBA) / wOBA_scale) * row['PA']


# def batting_runs(row):
#     lgR = 10000
#     PF = row['Team']
#     totPA = 32000
#     Al =
#     Nl =
#     return row['wRAA'] + (lgR / totPA  - (PF * lgR / totPA)) * totPA + (lgR / totPA - (AL or NL non-pitcher wRC/PA)) * totPA


#def get_PF(team):