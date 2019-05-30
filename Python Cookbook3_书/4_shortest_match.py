import re


def shortmch1(string):
    str_pat = re.compile(r'\"(.*)\"')   # 在正则表达式中 * 操作符采用的是贪婪策略，所以匹配过程是找出最长的可能来匹配
    return str_pat.findall(string)


def shortmch2(string):
    str_pat = re.compile(r'\"(.*?)\"')   # 在 * 操作符后加上 ? 修饰符就会以非贪心方式进行匹配
    return str_pat.findall(string)


def exsampl():
    text1 = 'Computer says "no."'
    text2 = 'Computer says "no." Phone says "yes."'
    print('text1: {t1}\ntext2: {t2}'.format(t1=text1, t2=text2))
    res1 = shortmch1(text1)
    res2 = shortmch1(text2)
    print('贪心模式进行匹配')
    print(res1)
    print(res2)

    res3 = shortmch2(text1)
    res4 = shortmch2(text2)
    print('非贪心模式进行匹配')
    print(res3)
    print(res4)


if __name__ == "__main__":
    exsampl()