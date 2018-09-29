import os, stat
from os import path
from flask import Flask, render_template, request, redirect, url_for
from  predict   import predictint, imageprepare, init_tf, predict



full_filename = 'static/outputSep-17-2018_2220-1537203051.png'

st = os.stat(full_filename)
#print(st)

init_tf()
imvalue = imageprepare(full_filename)

'''
predint = predictint(imvalue)
print (predint[0])  # first value in list
'''


print ( predict(imvalue))
print ( predict(imvalue))