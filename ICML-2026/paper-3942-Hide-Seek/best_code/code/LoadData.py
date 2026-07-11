"""
This program load and preprocessing data.
If there is no actual data import, a random data table will be generated.
"""
import pandas as pd
import numpy as np
import random


def load_data(initial_data):
    if initial_data:
        # 1. read data from real data set
        raw_data = initial_data
    else:
        # 2. create data
        raw_data = [random.randrange(1, 251, 1) for rows in range(10)]

    return raw_data

"""
# create list
def list_extend(_list,times):
    OrginList=[_list for _i in range(times)]
    
    _extendList=[]
    
    for k,j in enumerate(OrginList):
        for i in j:
            _extendList.append(i)
        
    return _extendList

# add time variable
def Date(start,end):
    _date=[]
    for i in range((end-start).days+1):
        _date.append(start + datetime.timedelta(days=i))
    return _date

# create data
ShopName=["Sam's Club",'bravo','WU MART','Vangguard']
start=datetime.date(2019,1,1)
end=datetime.date(2019,6,1)

data = pd.DataFrame({'ShopName':list_extend(ShopName,38),
                      'income':random.sample(list(range(0,200)),152),
                      'date':Date(start,end)})
data.head(3)

"""