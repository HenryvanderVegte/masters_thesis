import pylab
from classification.util.global_vars import *
import matplotlib.pyplot as plt
import os

a = [pow(10, i) for i in range(10)]

x = [1,2,4,8,16,32,64,128,256,512,989]

y = [34.46, 37.55, 41.37, 46.90, 49.42, 54.29, 56.42, 58.46, 59.89, 60.15, 61.31]

fig = plt.figure()
#ax = fig.add_subplot(2, 1, 1)

plt.xlabel("Features (Count)")
plt.ylabel("Accuracy (%)")
plt.xlim(1, 989)
plt.ylim(0, 100)
#plt.xscale('log', basex=2)

line, = plt.plot(x, y, color='blue', lw=2)

plt.savefig(os.path.join(ROOT_FOLDER, 'test//feature_plot3.png'))

