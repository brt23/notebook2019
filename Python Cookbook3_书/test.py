def no_space(x):
    while ' ' in x:
        x = x.replace(' ', '')
     
    return x


if __name__ == "__main__":
    print(no_space('8 j 8   mBliB8g  imjB8B8  jl  B'))