import math
import numpy as np
import matplotlib.pyplot as plt

def xcoord(t):
    return 16*( (math.sin(t ))**3) / 15.837

def ycoord(t):
    return  ( (13*math.cos(t)) - (5 * math.cos(2*t) ) - ( 2 * math.cos(3*t) ) - math.cos(4*t) ) / 15.837

xcoords = []
ycoords = []
for i in np.linspace(0, 2*math.pi, 20):
    x = xcoord(i)
    y = ycoord(i)
    plt.plot(x, y, '*')
    xcoords.append(x)
    ycoords.append(y)

plt.show()
print(max( max(xcoords), max(ycoords) ) )

for i in range(len(xcoords)):
    print("{}, {}, ".format(xcoords[i], ycoords[i]) )

    
