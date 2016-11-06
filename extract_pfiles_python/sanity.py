# to test if the TCL and Python script have the same output
import pandas as pd
import sys
print("loading 1")
arr1 = pd.read_csv(sys.argv[1], delim_whitespace=True, header=None).values
print("loading 2")
arr2 = pd.read_csv(sys.argv[2], delim_whitespace=True, header=None).values

(h, w) = arr1.shape
if arr1.shape != arr2.shape:
    print("warning: dimensions differ: {} != {}".format(arr1.shape, arr2.shape))
epsilon = 0.01
abs_epsilon = 1e-05
max_delta = 0.0

for col in range(0, h):
    for row in range(0, w):
        a = arr1[col][row]
        b = arr2[col][row]
        if abs(a - b) > abs_epsilon + epsilon * abs(b):
            if b != 0:
                relative_delta = abs(a - b) / abs(b)
                if relative_delta > max_delta:
                    max_delta = relative_delta
                    print("new max: {}".format(max_delta))
                if relative_delta > epsilon:
                    print("[{}][{}] does not match, a = {}, b = {}, delta = {}".format(col, row, a, b, relative_delta))
    if col % 100000 == 0:
        print(str(int(col / h * 100)) + "%")

print("max delta found: {}".format(max_delta))
