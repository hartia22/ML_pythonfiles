# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:32:42 2020

@author: Majic
"""

import pandas as pd
import os


os.listdir('e:\\\\' )

def function(paths):
    if(len(paths) == 1):
        return 1
    else:
        function(paths[1:len(paths)-1])
        return function(paths[1:len(paths)-1])

paths = os.listdir('d:\\\\' )
#print(paths[1:len(paths)-1])

print(function(os.listdir('d:\\\\' )))