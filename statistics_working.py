from __future__ import print_function
import numpy as np
import sys
sys.path.append('/Users/pawnoutlet/Documents/fdfault/data')
import fdfault
import seistools.coulomb
#from coulomb import coulomb_2d
import matplotlib.pyplot as plt

# defining some parameters to use it later

prob_name= 'seed'              # parameters related to problem
number_of_realization=2

receiver_fault_orientation= np.array([0,1])    # parameters related to coloumb stress
mu=0.4
coloumbs_stress_time_step= 16                       # Time step at which stresses are required  	     

#range_values=np.zeros([100,2])


distance_start= 15000.              # Since fault is at 20 km so I am taking 5 km distance on both sides
distance_end= 20000                 
space_interval=50                  # distance between two grid points 
distance_start2= 25000      # the second array will start from here

index_y_start= int(distance_start/space_interval)    # 
index_y_end= int(distance_end/space_interval)
index_y_start2=  int(distance_start2/space_interval)
range_of_iter=index_y_end-index_y_start +1
print(index_y_start, index_y_end, index_y_start2, range_of_iter)


distance_array= np.linspace( (distance_end-distance_start )/1000., 0, range_of_iter) 
total_dim_x = 801


cff_w_distanc1 = np.zeros(( range_of_iter,number_of_realization* total_dim_x) )   
cff_w_distanc2= np.zeros(( range_of_iter,number_of_realization* total_dim_x) )
rupture_length= np.zeros((number_of_realization))




string= 'sxxbody.' +'sxx'+'.shape[0]'   # to find the time step 



summ_switch= 0
start_x= 400
for kk in range(number_of_realization):
	# loading all the data for each realization
	problem_name= prob_name + str(kk) +'h' + '1' + 'RMS'+ '0_01' 
	sxxbody = fdfault.output(problem_name,'sxx'+'body')
	sxxbody.load()
	sxybody = fdfault.output(problem_name,'sxy'+'body')
	sxybody.load()
	syybody = fdfault.output(problem_name,'syy'+'body')
	syybody.load()

	# this part is for the patch length
	vfault = fdfault.output(problem_name,'v'+'fault')
	vfault.load()	
	

	print(problem_name)


	
	# Need to calculate the patch length that is rupturing
	
	fault_dimension= np.linspace(0,80., vfault.V.shape[1])
	rupt_indx= vfault.V.argmax(axis=1)
	rupt_slip_rate= vfault.V.max(axis=1)


	rupt_slip_test= vfault.V[coloumbs_stress_time_step, :]   # store all the values in this array
	print(rupt_slip_test.shape, 'the shape of new array')
	max_slip_side1= np.max(rupt_slip_test[rupt_indx[0] :vfault.V.shape[1] ])
	max_slip_side2= np.max(rupt_slip_test[0: rupt_indx[0]  ])
	indx_max_slip_side1 = np.where(rupt_slip_test== max_slip_side1)
	print('max slip', max_slip_side1, max_slip_side2, indx_max_slip_side1)

	if  max_slip_side1>= 10.0:
		indx_max_slip_side1 = np.where(max_slip_side1)
	else:
		for runs in range (coloumbs_stress_time_step-3):
		
			rupt_slip_test= vfault.V[coloumbs_stress_time_step- 1- runs, :]
			max_slip_side1= np.max(rupt_slip_test[rupt_indx[0] :vfault.V.shape[1] ])
			if  max_slip_side1>= 10.0:
				indx_max_slip_side1 = np.where(rupt_slip_test== max_slip_side1)
				break
			else:
			 	indx_max_slip_side1 = 	rupt_indx[0]	

	if  max_slip_side2>= 10.0:
		indx_max_slip_side2 = np.where(max_slip_side2)
	else:
		for runs in range (coloumbs_stress_time_step-3):
		
			rupt_slip_test= vfault.V[coloumbs_stress_time_step- 1- runs, :]
			max_slip_side2= np.max(rupt_slip_test[0:rupt_indx[0] ])
			if  max_slip_side2>= 10.0:
				indx_max_slip_side2 = np.where(rupt_slip_test== max_slip_side2)
				break
			else:
			 	indx_max_slip_side2 = 	rupt_indx[0]		

	rupture_length[kk] = fault_dimension[indx_max_slip_side1] - fault_dimension [indx_max_slip_side2]
	print('rupture length', rupture_length)
	#print('max slip', max_slip_side1, max_slip_side2)

	# if  rupt_slip_rate[coloumbs_stress_time_step]   >= 10.0:
	# 	rupture_length[kk]= fault_dimension [  rupt_indx[coloumbs_stress_time_step]  ]-  fault_dimension [ rupt_indx[0] ]
	# else:
	# 	for runs in range (coloumbs_stress_time_step-3):
	# 		if rupt_slip_rate [coloumbs_stress_time_step- 1- runs]   >= 10.0:
	# 			rupture_length[kk]= fault_dimension [  rupt_indx[coloumbs_stress_time_step-1- runs]  ]-  fault_dimension [ rupt_indx[0] ]
	# 			break
	# 		else:
	# 			rupture_length[kk] =0
					

	





	total_time= eval(string)
#	k= 2
	k= coloumbs_stress_time_step
	c_stress=seistools.coulomb.coulomb_2d(sxxbody.sxx[k,:,:], sxybody.sxy[k,:,:], syybody.syy[k,:,:], receiver_fault_orientation, mu)
	# coulomb_min= c_stress.min()
	# coulomb_min= c_stress.max()
	
#	print(c_stress.shape)


	
	for dim_y in range(range_of_iter):                 # populating the values of the stresses in a matrix, the matrix row is disntace and coloumn is value
		for dim_x in range(total_dim_x):
			cff_w_distanc1[dim_y, summ_switch* total_dim_x +dim_x]= c_stress[start_x+ dim_x, dim_y+index_y_start]    # one row is for one distance and the second row is for second disntace
			cff_w_distanc2[dim_y, summ_switch* total_dim_x +dim_x]= c_stress[start_x+ dim_x, index_y_start2-dim_y]
		
		
		

	summ_switch= summ_switch +1  	       

cff_full= np.concatenate((cff_w_distanc1, cff_w_distanc2), axis=1)
#print(cff_full.shape)
#print(cff_w_distanc1.shape)

print(cff_full.shape)
print(cff_w_distanc1.shape)
#print(cff_w_distanc1[0,:])
#np.savetxt('statistics_stress.out', cff_w_distanc1 , delimiter=',')

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



#plt.colorbar()

#	plt.xlabel('Position along the fault (km)', fontsize=16, color='black')
#	plt.ylabel('Position accross the strike (km)', fontsize=16, color='black')



	







