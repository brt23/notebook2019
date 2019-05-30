"""
模仿了内置函数 sorted()、min()以及 max()对 key 函数的使用方式
这个函数的功能是去重复不改变元素位置
"""

def dedupe(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val in seen:
            yield item  # yield 后面的代码是会执行的
            seen.add(val)