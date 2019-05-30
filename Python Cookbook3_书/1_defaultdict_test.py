"""
展示 defaultdict 的特点
使用 defaultdict 可以使代码更容易理解
"""


from collections import defaultdict


def show_defaultdict(seq):
    tmp = defaultdict(list) # defaultdict 在初始化时，可以设定字典元素是什么数据类型，这里初始化的是 list 类型
    for index, iseq in enumerate(seq):
        tmp[iseq].append(index)

    return tmp


if __name__ == "__main__":
    test_seq = ['2', '1', '2', '3', '4', '4']
    res = show_defaultdict(test_seq)
    print(res)