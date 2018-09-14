import os, stat
from os import path
from flask import Flask, render_template, request, redirect, url_for

sfile = 'static/car.jpg'
st = os.stat(sfile)
print(st)

full_filename = 'images/car.jpg'
full_filename = 'images'
st = os.stat(full_filename)
print(st)

path = 'images'
os.chmod(path, 0o777)

os.chmod(path, st.st_mode | stat.S_IWOTH)

st = os.stat('uploads')
os.chmod(path, st.st_mode | stat.S_IWOTH)
# os.chmod('upload', stat.st_mode | stat.S_IWOTH)

print('\n os.access(full_filename, os.X_OK) ')
print ( os.access(path, os.X_OK) ) # Check for read access
