import ctypes


_mod = ctypes.cdll.LoadLibrary('/home/y/文档/CodeHub/Python_Practice/Python Cookbook3/libgcd.so')

gcd = _mod.gcd
gcd.argtypes = (ctypes.c_int, ctypes.c_int)
gcd.restype = ctypes.c_int


res = gcd(15, 6)
print(res)