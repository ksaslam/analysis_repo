from __future__ import print_function
import numpy as np
import sys
sys.path.append('/Users/pawnoutlet/Documents/fdfault/data')
import fdfault
#import seistools.coulomb
import seistools.coulomb
import matplotlib.pyplot as plt

# defining some parameters to use it later

prob_name= 'seed'              # parameters related to problem
number_of_realization=2

receiver_fault_orientation= np.array([0,1])    # parameters related to coloumb stress
mu=0.4
coloumbs_stress_time_step= 16                       # Time step at which stresses are required  	     

#range_values=np.zeros([100,2])


              # Since fault is at 20 km so I am taking 5 km distance on both sides
#distance_start= 20000                 
distance_start= 15000
distance_end= 20000  
space_interval=50
#index_y_start= int(distance_start/space_interval)-10    # 10 grid points away from the fault 
index_y_start= int(distance_start/space_interval)
index_y_end= int(distance_end/space_interval)-10
range_of_iter=index_y_end-index_y_start +1
#range_of_iter= 1
total_dim_x = 801

distance_array= np.linspace( 5000, 500, range_of_iter )
print(distance_array) 

#cff_w_distanc1 = np.zeros(( number_of_realization, total_dim_x) )   

string= 'sxxbody.' +'sxx'+'.shape[0]'   # to find the time step 




start_x= 400

positive_zones= np.zeros((range_of_iter, number_of_realization))
negative_zones=np.zeros((range_of_iter,  number_of_realization))
each_pos_zone_len= np.zeros((number_of_realization, range_of_iter, 50))
each_neg_zone_len= np.zeros((number_of_realization, range_of_iter, 50))


for kk in range(number_of_realization):
	# loading all the data for each realization
	problem_name= prob_name + str(kk) +'h' + '1' + 'RMS'+ '0_01' 
	sxxbody = fdfault.output(problem_name,'sxx'+'body')
	sxxbody.load()
	sxybody = fdfault.output(problem_name,'sxy'+'body')
	sxybody.load()
	syybody = fdfault.output(problem_name,'syy'+'body')
	syybody.load()
	print(problem_name)
	
	total_time= eval(string)
#	k= 2
	k= coloumbs_stress_time_step
	c_stress=seistools.coulomb.coulomb_2d(sxxbody.sxx[k,:,:], sxybody.sxy[k,:,:], syybody.syy[k,:,:], receiver_fault_orientation, mu)
	# coulomb_min= c_stress.min()
	# coulomb_min= c_stress.max()
	
#	print(c_stress.shape
	for dim_y in range(range_of_iter):	
		for dim_x in range(total_dim_x):

			if c_stress[start_x+ dim_x, index_y_start + dim_y ]>= 0 and c_stress[start_x+ dim_x+1, index_y_start + dim_y ] < 0 :
									
					if  c_stress[start_x+ dim_x+2, index_y_start + dim_y] <0  and c_stress[start_x+ dim_x+3, index_y_start + dim_y] <0 and c_stress[start_x+ dim_x+4, index_y_start+ dim_y]<0 and c_stress[start_x+ dim_x+5, index_y_start + dim_y] <0 and c_stress[start_x+ dim_x+6, index_y_start + dim_y] <0 and c_stress[start_x+ dim_x+7, index_y_start + dim_y] <0 and c_stress[start_x+ dim_x+8, index_y_start + dim_y] and c_stress[start_x+ dim_x+9, index_y_start + dim_y] and c_stress[start_x+ dim_x+10, index_y_start + dim_y] <0:

						negative_zones[dim_y, kk] = negative_zones[dim_y, kk] +1
						
						neg_zone= 0
						for runs in range (100):
							if c_stress [start_x+ dim_x+1+runs , index_y_start + dim_y] < 0:
								neg_zone= neg_zone +1
								neg_zone_len= neg_zone * 50.
								each_neg_zone_len[kk, dim_y ,int( negative_zones[dim_y,kk]-1 )] = neg_zone_len

							else:
								break	
						
								




			if c_stress[start_x+ dim_x, index_y_start + dim_y]< 0 and c_stress[start_x+ dim_x+1, index_y_start + dim_y] >= 0 :
									
					if  c_stress[start_x+ dim_x+2, index_y_start + dim_y] >=0  and c_stress[start_x+ dim_x+3, index_y_start + dim_y] >=0 and c_stress[start_x+ dim_x+4, index_y_start + dim_y] >=0 and c_stress[start_x+ dim_x+5, index_y_start + dim_y] >=0 and c_stress[start_x+ dim_x+6, index_y_start + dim_y] >=0 and c_stress[start_x+ dim_x+7, index_y_start + dim_y] >=0 and c_stress[start_x+ dim_x+8, index_y_start + dim_y] >=0 and c_stress[start_x+ dim_x+9, index_y_start + dim_y] >=0 and c_stress[start_x+ dim_x+10, index_y_start + dim_y] >=0:
			
						positive_zones[dim_y, kk] = positive_zones[dim_y, kk] +1

						pos_zone= 0
						for runs in range (100):
							if c_stress [start_x+ dim_x+1+runs , index_y_start + dim_y] > 0:
								pos_zone= pos_zone +1
								pos_zone_len= pos_zone * 50.
								each_pos_zone_len[kk,dim_y ,int( positive_zones[dim_y, kk]-1 )] = pos_zone_len
							else:
								break			
															



print(negative_zones[range_of_iter-1,:])
print( each_neg_zone_len[0, range_of_iter-1 , :] )
print(each_neg_zone_len.shape)


full_pos_len_zone_matrix=np.zeros((range_of_iter, number_of_realization*50 +1 ) )
for jj in range( range_of_iter):
	b= np.zeros((1))
	for ii in range (number_of_realization):
		a= each_pos_zone_len[ii, jj, :]
		b= np.concatenate([a, b])             # this is just to concatenate all the values b, a are dummy
	full_pos_len_zone_matrix[jj, :]= b	

plt.plot(distance_array, full_pos_len_zone_matrix[:, :], 'o')
#plt.axis([-1, 5, -30, 30])
plt.show()

#print(a.shape, b.shape, 'these are shapes of all ')	


#plt.hist(positive_zones[0, :], bins= 15, facecolor='yellow', edgecolor='gray', label='Positive zones')
#plt.hist(negative_zones[0, :], bins= 15, facecolor='red', edgecolor='gray', label='negative zones')
#plt.xlabel(' # of zones')
#plt.ylabel('# per bin')
#plt.title('Pdf of stress zones', fontsize=16,color='black')
#plt.legend( loc='upper right')
#plt.show()

#print(cff_full.shape)
#print(cff_w_distanc1.shape)



#plt.hist(cff_w_distanc1 [0,total_dim_x:2*total_dim_x], bins=80)
#plt.hist(cff_full[0,2*total_dim_x:3*total_dim_x], bins=80)
#plt.show()


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

# plt.pcolor(c_stress[start_x:start_x+ total_dim_x, :], vmin= -10, vmax= 10)
# plt.colorbar()
# plt.show()

#plt.colorbar()

#	plt.xlabel('Position along the fault (km)', fontsize=16, color='black')
#	plt.ylabel('Position accross the strike (km)', fontsize=16, color='black')



	







