# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:49:57 2020

@author: Majic
"""

import re

urls = r'''
https://www.google.com
http://yahoo.com
https://www.whitehouse.gov
http://craigslist.org
'''

pater = re.compile(r'https?://(www\.)?\w+\.\w+')
pater = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
mat = pater.finditer(urls)
for m in mat:
    print(m.group(0))