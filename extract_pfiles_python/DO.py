#!/usr/bin/env python3
import subprocess
from pprint import pprint

db = "../ears2/db/train/all240302"
context = 10
type = "train"
convIdFile = "../data/conversations." + type

args = [
    "python3",
    "./readDB.py",
    db,
    "context{context}-{type}".format(**locals()),
    convIdFile,
    "-dataPath", "out/context{context}".format(**locals()),
    "-delta", str(context)
]
print("calling")
pprint(args)
subprocess.call(args)
