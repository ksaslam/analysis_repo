# example using output class in python

# required arguments are problem name and output unit name
# data directory is optional, if no argument provided assumes it is the current working directory

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/pawnoutlet/Documents/fdfault/data')
import fdfault
import matplotlib.backends.backend_pdf
import seistools.coulomb

# ________________________________________

# Specify the prob name and component




#prob_name= 'bi_100_42'
prob_name= 'seed1'

receiver_fault_orientation= np.array([0,1])

mu=0.4

rms_ratio= np.array([10**-2])
#hurst= np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])
hurst= np.array([1.0])


for ii in range(len(rms_ratio)):
  for jj in range(len(hurst)):
    oldstr_hurst= str(hurst[jj])
    oldstr_rms= str(rms_ratio[ii])
    newstr_rms = oldstr_rms.replace(".", "_")
    newstr_hurst = oldstr_hurst.replace(".", "_")
#    problem_name= prob_name + 'h' + newstr_hurst + 'RMS'+ newstr_rms 
    problem_name= prob_name + 'h' + '1' + 'RMS'+ newstr_rms 
    sxxbody = fdfault.output(problem_name,'sxx'+'body')
    sxxbody.load()
    sxybody = fdfault.output(problem_name,'sxy'+'body')
    sxybody.load()
    syybody = fdfault.output(problem_name,'syy'+'body')
    syybody.load()    
    #print(sxxbody)
    string= 'sxxbody.' +'sxx'+'.shape[0]'
    #print(string)
    total_time= eval(string)
    #print(total_time)
     
# ____________________________________


  
    for kk in range(1):
    #for k in range(total_time):
    #   print(k)
      
      #station_loc = station_location[i]
      #station_loc= 'x10y35'
#    k=total_time-1
#    k=1
      k= kk+16

      title=  'hurst'+newstr_hurst +'rms' + newstr_rms+  'coulomb_Stress '   

      c_stress=seistools.coulomb.coulomb_2d(sxxbody.sxx[k,:,:] , sxybody.sxy[k,:,:], syybody.syy[k,:,:], receiver_fault_orientation, mu)
#      plt.pcolor(c_stress[600:800,300:500], cmap='RdBu', vmin=-20, vmax=20)            
#      plt.colorbar()
#      plt.show()
#      plt.figure(k)
#      plt.pcolor(sxxbody.x, sxxbody.y, c_stress[:,:], cmap='RdBu', vmin=-20, vmax=20)
  # #    plt.pcolor(sxxbody.x, sxxbody.y, c_stress[:,:]/c_stress.max(), cmap='RdBu', vmin=-1, vmax=1)
  # #    plt.pcolor(sxxbody.x, sxxbody.y, sxxbody.sxx[k,:,:]+100, cmap='RdBu')
  # #    plt.axis([32, 55, 15, 25])
#      plt.colorbar()
  #     plt.title(title,fontsize=16, color='black')
  #     plt.xlabel('Position along the fault (km)', fontsize=16, color='black')
  #     plt.ylabel('Position accross the strike (km)', fontsize=16, color='black')
  #     plt.savefig(str(k) +'.png')
#      plt.show()
        

plot_stress=c_stress[400:1202,300:502]
plt.figure(2)
plt.hist(plot_stress[:,0], bins=80)
plt.show()


      # plt.figure(2+i)
      # plt.grid(True)
      # plt.plot(vxbody.t, vxbody.vx,'r',label='Patch')
      # plt.plot(vxbody.t, vxbody1.vx,'b',label='Single point')
      # plt.plot(vxbody.t, vxbody2.vx,'g',label='Transient_SW')
      # plt.plot(vxbody.t, vxbody3.vx,'black',label='Transient_single')
      # plt.xlabel('X');
      # plt.ylabel('Velocity')
      # plt.title(title, fontsize=16,color='black')
      # plt.legend( loc='upper left')
      # plt.axis([0,4,-.1,.1])

# pdf = matplotlib.backends.backend_pdf.PdfPages('uni_100_42'+'_x_seismogram.pdf')
# for fig in range(1, total_time+1): ## will open an empty extra figure :(
#   pdf.savefig( fig )
# pdf.close()
    #plt.show()


    # sec

    #
    #plt.pcolor(vxbody.x, vxbody.y, vxbody.sxy[time_step,:,:])
    #plt.axis([0, 24, 0, 32])
    #plt.colorbar()
    #plt.title('Particle velocity- fault parallel(m/s)',fontsize=16, color='black')
    #plt.xlabel('Position across the fault (km)', fontsize=16, color='black')
    #plt.ylabel('Position along the strike (km)', fontsize=16, color='black')
    #plt.savefig('foo.png')
    #plt.show()

