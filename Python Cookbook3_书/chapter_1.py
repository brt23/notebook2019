# -*- coding: utf-8 -*-
"""
Spyder Editor Y

This is a temporary script file.
"""

from collections import deque
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# 1.3 保存最后 N 个元素
# -----------------------------------------------------------------------------
def search(lines, pattern, history=5):
    previous_lines = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line, previous_lines
        previous_lines.append(line)
        
with open('/home/y/information.txt', 'r') as f:
    for line, previous_lines in search(f, 'CXX', 5):
        for pline in previous_lines:
            print(pline, end='')
        print(line, end='')
        print('-' * 20)
# -----------------------------------------------------------------------------
# 1.9 在两个字典中寻找相同点
# -----------------------------------------------------------------------------
a = {'x': 1, 'y': 2, 'z': 3}
b = {'w': 10, 'x': 11, 'y': 2}
a.keys() & b.keys()
a.keys() - b.keys()
a.items() & b.items()
c = {key: a[key] for key in a.keys() - {'z', 'w'}}
c
# -----------------------------------------------------------------------------
# 1.10 从序列中移除重复项且保持元素间顺序不变
# -----------------------------------------------------------------------------
def dedupe(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(item)
            
a = [1, 5, 2, 1, 9, 1, 5, 10]
list(dedupe(a))
# -----------------------------------------------------------------------------
# 1.10 
# -----------------------------------------------------------------------------
words = [
'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
'my', 'eyes', "you're", 'under'
]
from collections import Counter
word_counts = Counter(words)
top_three = word_counts.most_common(3)
print(top_three)
word_counts['not']
# -----------------------------------------------------------------------------
# 1.15 根据字段将记录分组
# ----------------------------------------------------------------------------
from operator import itemgetter
from itertools import groupby

rows = [
{'address':
 '5412 N CLARK', 'date': '07/01/2012'},
{'address':
 '5148 N CLARK', 'date': '07/04/2012'},
{'address':
 '5800 E 58TH', 'date': '07/02/2012'},
{'address':
 '2122 N CLARK', 'date': '07/03/2012'},
{'address':
 '5645 N RAVENSWOOD', 'date': '07/02/2012'},
{'address':
 '1060 W ADDISON', 'date': '07/02/2012'},
{'address':
 '4801 N BROADWAY', 'date': '07/01/2012'},
{'address':
 '1039 W GRANVILLE', 'date': '07/04/2012'},
]
    
rows.sort(key=itemgetter('date'))

for date, items in groupby(rows, key=itemgetter('date')):
    print(date)
    for i in items:
        print(' ', i)





















