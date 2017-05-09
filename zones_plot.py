from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


positive_zones= np.loadtxt('positive_zones.out',delimiter=',')
negative_zones= np.loadtxt('negative_zones.out',delimiter=',')
each_pos_zone_len= np.loadtxt('each_postv_zone_len.out',delimiter=',')
each_neg_zone_len= np.loadtxt('each_negtv_zone_len.out',delimiter=',')


plt.hist(positive_zones, bins= 15, facecolor='yellow', edgecolor='gray', label='Positive zones')
plt.hist(negative_zones, bins= 15, facecolor='red', edgecolor='gray', label='negative zones')
plt.xlabel(' # of zones')
plt.ylabel('# per bin')
plt.title('Pdf of stress zones', fontsize=16,color='black')
plt.legend( loc='upper right')
plt.show()

each_neg_zone_array=np.trim_zeros(each_neg_zone_len[0,:])


ax = plt.figure().gca()
ax.plot( each_neg_zone_array,'o' )
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()


