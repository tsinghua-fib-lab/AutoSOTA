# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:19:22 2020

This file defines a function which we use to generate directories if they do not
already exist. The purpose is simply to make the code more reproducible in that
you can run it on a different computer which doesn't have the folders we use
for organizing the output files, and the folders will be created to store
all the output in an organized fashion. 

"""
from pathlib import Path
def make_dirs(dirs):
    if type(dirs)==str:
        dirs = [dirs]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)