# to test if the TCL and Python script have the same output
import pandas as pd

print("loading 1")
arr1 = pd.read_csv("pyoutput/context10-train-BC.txt", delim_whitespace=True, header=None).values
print("loading 2")
arr2 = pd.read_csv("tcloutput/context10-train-BC.txt", delim_whitespace=True, header=None).values
print("subtra")

(h, w) = arr1.shape
epsilon = 0.01
max_delta = 0
for col in range(0, h):
    for row in range(0, w):
        a = arr1[col][row]
        b = arr2[col][row]
        relative_delta = abs(a-b)/abs(b)
        max_delta = max(relative_delta, max_delta)
        if relative_delta > epsilon:
            print("[{}][{}] does not match, delta = {}".format(col, row, relative_delta))
    if col % 1000 == 0:
        print(str(int(col / h * 100)) + "%")

print("max delta found: {}".format(max_delta))