import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import os


file_name1 = 'serial_code.txt'
path1 = os.path.abspath(file_name1)
data_time1 = np.genfromtxt(path1, delimiter=',')
plt.axis('off')
plt.imshow(-data_time1, cmap='Greys', interpolation=None)
png_file = 'serial_ray_tracing_object.png'
plt.savefig(png_file)