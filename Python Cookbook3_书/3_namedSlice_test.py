"""
使用命名切片让代码更加灵活
"""

# 使用切片对象和直接使用切片一样
items = [0, 1, 2, 3, 4, 5, 6]
a = slice(2, 4)
print(items[2:4])
print(items[a])

# 切片对象有三个属性，分别是起始，结束，步长
a = slice(10, 50, 2)
print(a.start)
print(a.stop)
print(a.step)

# 当切片 indices 方法把切片映射到特定大小序列上，会返回一个元组，所有值都在序列的边界内，可以避免 indexerror
a = slice(0, 50, 1)
s = 'HelloWorld'
a.indices(len(s))
print(a)
print(*a.indices(len(s)))
for i in range(*a.indices(len(s))):
    print(s[i])