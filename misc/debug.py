"""
Because pycharm can't run in debug mode when running code as a module, this is a tiny wrapper script.

https://stackoverflow.com/questions/28955520/intellij-pycharm-cant-debug-python-modules

Usage example:

python misc/debug.py trainNN.train configs/bla/bla.jsonn

"""
import sys
import os
import runpy

path = os.path.dirname(sys.modules[__name__].__file__)
path = os.path.join(path, '..')
sys.path.insert(0, path)
del sys.argv[0]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
