#!/usr/bin/env python
from __future__ import print_function
f=open("semeion.data")
ll=f.readlines()
n=len(ll)
w=16
h=16
ww=w
hh=n*h
print("P3")
print("%d %d" % (ww, hh))
print("%d" % 1)
for l in ll:
	x = y = 0
	for p in l.split(' '):
		if x>=w:
			x = 0
			y+=1
		if y>=h:
			break
		r = g = b = float(p)
		print(" %d %d %d" % (r, g, b), end="")
		x+=1
	print("")
