import re


text1 = '/* this is a comment */'
text2 = '''/* this is a
              multiline comment */
        '''

comment = re.compile(r'/\*(.*?)\*/')
res1 = comment.findall(text1)
res2 = comment.findall(text2)
print(res1)
print(res2)

comment1 = re.compile(r'/\*((?:.|\n)*?)\*/')
res3 = comment1.findall(text2)
print(res3)