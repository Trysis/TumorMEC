#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a feature as a map froma given dataframe
Ex: 
    m=ExtractMap(df,"Density20")
    plt.imshow(m)
    
    
Created on Wed Sep 7 11:50:57 2022

@author: paolo pierobon
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ExtractMap(df,featName):
    x=np.int32(np.array((df['X']-df['X'].min())/40))
    y=np.int32(np.array((df['Y']-df['Y'].min())/40))
    m=np.zeros([y.max()+1,x.max()+1]);
    feat=np.array(df[featName])
    npoints=df.shape[0]
    for i in range(npoints):
        #   m[y[i],x[i]]=fiber[i]
        m[y[i],x[i]]=feat[i];
    return m;
