def dotproduct(va, vb):
    o = 0
    for a, b in zip(va, vb):
        o += a * b
    return o


x = dotproduct([1, 2, 3], [2, 3, 4])
print(x)
