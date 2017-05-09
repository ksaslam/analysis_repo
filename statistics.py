from __future__ import print_function
import numpy as np
import sys
sys.path.append('/Users/pawnoutlet/Documents/fdfault/data')
import fdfault
import seistools.coulomb
import matplotlib.pyplot as plt



cff_w_distanc1= np.loadtxt('statistics_stress.out',delimiter=',')
print(cff_w_distanc1.shape)
total_dim_x= 1602
histogram= np.ones((101,80))
distance_array= np.linspace( 5, 0,cff_w_distanc1.shape[0] ) 
print(cff_w_distanc1.min(), cff_w_distanc1.max() )
rupture_leng= np.loadtxt('rupture_length.out',delimiter=',')


plt.hist(cff_w_distanc1 [0,0:801], bins=30, facecolor='red', edgecolor='gray', label= ' distance - 5 km')
plt.xlabel(' stress')
plt.ylabel('# per bin')
plt.title('Pdf of col. stress values - 5 km', fontsize=16,color='black')
plt.legend( loc='upper right')
plt.show()

plt.hist(cff_w_distanc1 [95,0:801], bins=30, facecolor='red', edgecolor='gray', label= 'col. stress close to fault km')
plt.xlabel(' stress')
plt.ylabel('# per bin')
plt.title('Pdf of col. stress values', fontsize=16,color='black')
plt.legend( loc='upper right')
plt.show()

#plt.show()
#plt.hist(cff_full[0,2*total_dim_x:3*total_dim_x], bins=80)
#plt.show()


max_stress= 20
#max_stress= cff_w_distanc1.max()
min_stress=-50
#min_stress= cff_w_distanc1.min()
for ii in range(101):
	for jj in range(cff_w_distanc1.shape[1]):
		if cff_w_distanc1[ii,jj] > max_stress:
			cff_w_distanc1[ii,jj] = max_stress
		if cff_w_distanc1[ii,jj] < min_stress:
			cff_w_distanc1[ii,jj] = min_stress

cff_w_distanc1[:,0]= min_stress
cff_w_distanc1[:,cff_w_distanc1.shape[1]-1]= max_stress

#plt.hist(cff_w_distanc1 [0,0:total_dim_x], bins=80)
#plt.show()

for ii in range(101):

 hist, edges = np.histogram(cff_w_distanc1[ii,:], bins=80)
 histogram[ii,:]= hist/max(hist) 
# # 	plt.plot(hist)
# # 	plt.show()
# # #histogram
# #    plt.imshow(cff_w_distanc1, cmap='RdBu', interpolation = 'bilinear')

# #print ('data =', data )
# #print ('histogram =', hist/ max(hist))
# #print ('edges =', edges)

x,y= np.meshgrid(distance_array, edges[1:len(edges)-1], indexing='ij')
plt.pcolor(x,y, histogram[:,1:cff_w_distanc1.shape[1]-1])
# print( hist.shape, edges.shape)
cbar = plt.colorbar()
#plt.colorbar()
cbar.set_label(' PDF of CFF', rotation=90)
#cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4','0.5','0.6','0.7','0.8','0.9','1.0'])
plt.xlabel('Distance_accross fault (km)', fontsize=16, color='black')
plt.ylabel('coulomb failure function (Mpa)', fontsize=16, color='black') 
plt.ylabel('coulomb failure function (Mpa)', fontsize=16, color='black') 
plt.title('coulomb failure function with distance ', fontsize=16, color='black') 
plt.show()


# for xe, ye in zip(distance_array, cff_full):
    
# #    plt.scatter([xe] * len(ye), ye, c= ye , cmap='viridis')
#     plt.scatter([xe] * len(ye), ye, facecolor='0.5')
#plt.colorbar()
#plt.axis([0, 3, -80, 80])
# plt.show()

#plt.plot(distance_array, cff_w_distanc1 [:, 0:total_dim_x], 'o')
#plt.axis([-1, 5, -30, 30])
#plt.show()


#plt.hist(cff_full [100, :], bins=50 )
#plt.hist(cff_full [100, :])




# range_plot= int(range_of_iter/3.0) +1
# for ii in range(range_plot):
# 	plt.figure(ii)
# 	i= ii*2+1
# #	print(cff_full [ii, 0:2*total_dim_x])
# 	plt.hist(cff_full [i, 0:2*total_dim_x], bins=100 )
# 	plt.savefig(str(ii) +'.png') 



#plt.colorbar()

#	plt.xlabel('Position along the fault (km)', fontsize=16, color='black')
#	plt.ylabel('Position accross the strike (km)', fontsize=16, color='black')



	







