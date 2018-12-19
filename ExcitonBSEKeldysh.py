import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import numba
import tbmodels
from scipy.special import struve
from scipy.special import yn
import os
from scipy.linalg import eigh
import h5py

@numba.njit(parallel=True,nogil=True,fastmath=True)
def FastExcitonHamiltonian(H,E,shift,D,D_,W,cb,vb,Q,indexes):
    NH=H.shape[0];N=E.shape[0]
    for indx1 in numba.prange(NH):
        kx1,ky1,c1,v1=indexes[indx1]
       
        H[indx1,indx1]+=E[(kx1+Q[0])%N,(ky1+Q[1])%N,c1]-E[kx1,ky1,v1]+shift
        
        for indx2 in range(indx1,NH):
            kx2,ky2,c2,v2=indexes[indx2]
            
            O12=np.dot(D_[(kx1+Q[0])%N,(ky1+Q[1])%N].T,D[(kx2+Q[0])%N,(ky2+Q[1])%N]);
            O21=np.dot(D_[kx2,ky2].T,D[kx1,ky1]);
            H[indx1,indx2]-=W[(kx1-kx2)%N,(ky1-ky2)%N]*O12[c1,c2]*O21[v2,v1]
            #exchange
            
            O11=np.dot(D_[(kx1+Q[0])%N,(ky1+Q[1])%N].T,D[kx1,ky1]);
            O22=np.dot(D_[kx2,ky2].T,D[(kx2+Q[0])%N,(ky2+Q[1])%N]);
            H[indx1,indx2]+=W[Q[0],Q[1]]*O11[c1,v1]*O22[v2,c2]
            
            H[indx2,indx1]=H[indx1,indx2]
                                    
                                 
    return H

class ExcitonMoS2(object):

    def __init__(self,N,vb,cb,epsilon,r0=33.875,shift=0.5,spin_orbit=True):
        if spin_orbit:
            self.model =tbmodels.Model.from_wannier_files(
                                                            hr_file='TB-Models/MoS2_hr.dat',
                                                            wsvec_file='TB-Models/MoS2_wsvec.dat',
                                                            xyz_file='TB-Models/MoS2_centres.xyz',
                                                            win_file='TB-Models/MoS2.win'
                                                        )
        else:
            self.model =tbmodels.Model.from_wannier_files(
                                                            hr_file='TB-Models/wann_hr.dat',
                                                            wsvec_file='TB-Models/wann_wsvec.dat',
                                                            xyz_file='TB-Models/wann_centres.xyz',
                                                            win_file='TB-Models/wann.win'
                                                        )
        self.k1=self.model.reciprocal_lattice[0]
        self.k2=self.model.reciprocal_lattice[1]
        self.a1=self.model.uc[0]
        self.a2=self.model.uc[1]
        self.V=np.sqrt(3)/2*self.a1[0]**2
        self.norb=self.model.hamilton([0.,0.,0.]).shape[0]
        
        self.epsilon=epsilon #
        self.r0= r0/epsilon 
        self.shift=shift # scissor operator
        self.N=N # k-points grid
        self.vb=vb # valence bands
        self.cb=cb # conduction bands
        self.nv=len(self.vb);self.nc=len(self.cb)
        self.NH=self.N**2*self.nc*self.nv
        print('Exciton Hamiltonian size: '+str(self.NH)+'X'+str(self.NH))
        

        self.E=np.zeros((N,N,self.norb));self.K=np.zeros((N,N,3));
        self.D=np.zeros((N,N,self.norb,self.norb),dtype=np.complex)
        self.H=np.zeros((N,N,self.norb,self.norb),dtype=np.complex)
        for i,j in product(range(N),range(N)):
            self.H[i,j]=self.model.hamilton([i/N,j/N,0])
            e,d=np.linalg.eigh(self.H[i,j])
            self.E[i,j]=e;self.D[i,j]=d
            self.K[i,j]=i*self.k1/N+j*self.k2/N
        self.E_gap=np.min(self.E[:,:,cb[0]]-self.E[:,:,vb[-1]])+shift

        R=np.zeros((N,N,3))
        for i,j in product(range(N),range(N)):
            R[i,j]=self.a1*(i-N/2)+self.a2*(j-N/2)
        self.R=R
        WR=np.zeros((N,N))
        for i,j in product(range(N),range(N)):
            radius=np.linalg.norm(R[i,j])
            if radius!=0:WR[i,j]=14.4*np.pi/(2*self.epsilon*self.r0)*(struve(0,radius/self.r0)-yn(0,radius/self.r0))
            else:WR[i,j]=0
            

        WR=np.fft.fftshift(WR);self.W=np.fft.fft2(WR)/N**2
        self.indexes=[(kx,ky,i,j) for kx,ky,i,j in product(range(self.N),range(self.N),self.cb,self.vb)]
        

    def plotBandStructure(self,emax=5,emin=-8,E_Fermi=0):
        gamma=np.array([0.,0.,0.])
        M=np.array([0,0.5,0.])
        K=np.array([0.33,0.33,0.])
        kpoints=[];N1=20;N2=int(N1/2);N3=int(np.sqrt(N1**2+N2**2))
        for i in range(N1):
            kpoints.append(i*M/N1+(N1-i)*gamma/N1)
        for i in range(N2):
            kpoints.append(i*K/N2+(N2-i)*M/N2)
        for i in range(N3):
            kpoints.append(i*gamma/N3+(N3-i)*K/N3)
        kpoints=np.array(kpoints)

        E=[];D=[]
        for i in range(kpoints.shape[0]):
            e,d=np.linalg.eigh(self.model.hamilton(kpoints[i])-np.eye(self.norb)*E_Fermi)
            E.append(e);D.append(d)
        E=np.array(E);D=np.array(D)

        plt.figure()
        for i in range(self.norb):
            plt.plot(E[:,i],'-',label=str(i))
        # plt.legend()
        plt.plot(np.linspace(0,300),np.zeros(50),'--',color='black')
        plt.plot(np.ones(50)*(N1),np.linspace(emin,emax),'-',color='green')
        plt.plot(np.ones(50)*(N1+N2),np.linspace(emin,emax),'-',color='green')
        plt.xticks([0,N1,N1+N2,N1+N2+N3-1],['$\Gamma$','M','K','$\Gamma$'])
        plt.xlim([0.,N1+N2+N3-1])
        plt.ylim([emin,emax])

    def constuctExcitonHamiltonian(self,Q=[0,0,0]):
        self.Q=Q
        HH=np.zeros((self.NH,self.NH),dtype=np.complex)
        print('Construction Begin!')
        self.HH=FastExcitonHamiltonian(HH,self.E,self.shift,self.D,self.D.conj(),self.W,self.cb,self.vb,self.Q,self.indexes)
        print('Construction Done!')
        
    def solveExcitonHamiltonian(self,save=False):
        
        self.EH,self.DH=eigh(self.HH)
        self.DH=self.DH[:,np.argsort(self.EH)]
        self.EH=np.sort(self.EH)
        print('Hamiltonian Solved!')

        if save:
            if not os.path.exists('save'):
                os.makedirs('save')
            print('Saving Eigenvalues and Eigenfunctions...')
            
            h5f = h5py.File('save/'+str(self.Q)+'.h5', 'w')
            h5f.create_dataset('E', data=self.EH)
#             h5f.create_dataset('H', data=self.HH)
            h5f.create_dataset('D', data=self.DH)
            h5f.close()
            
            print('Saved!')

    def ExcitonWaveFunction(self,i,type='rho'):
        
        wave_k=self.DH[:,i].reshape(self.N,self.N,self.nc,self.nv)
        wave_k=np.sum(wave_k,axis=(2,3))
        wave_k/=np.sqrt(np.sum(wave_k.conj()*wave_k))
        
        wave_r=np.fft.fftshift(np.fft.fft2(wave_k))/self.N**2
        
        rho_k=wave_k.conj()*wave_k
        rho_r=wave_r.conj()*wave_r
        if type=='rho':
            return rho_k.real,rho_r.real
        elif type=='wave':
            return wave_k,wave_r
    
    def absobtionSpectrum(self,theta_array=[0],eta=0.03,omega_max=5,omega_n=5000):
        
        plt.figure()
        for theta in theta_array:
            P1=np.gradient(self.H,axis=0);P2=np.gradient(self.H,axis=1)
            Px=(np.sqrt(3)/2.*P1);Py=(P2+P1/2.)
            self.P=np.cos(theta)*Px+np.sin(theta)*Py
            for i in range(self.N):
                for j in range(self.N):
                    self.P[i,j]=np.linalg.multi_dot([self.D[i,j].conj().T,self.P[i,j],self.D[i,j]])
            P=np.array([self.P[indx] for indx in self.indexes])
            optical=np.zeros(self.NH)
            for i in range(self.NH):
                optical[i]=np.abs(np.sum(P*self.DH[:,i]))**2
            omega=np.linspace(0,omega_max,omega_n+1);omega=np.delete(omega,0)
            absorbtion=np.zeros(omega.size,dtype=np.complex)
            for i in range(omega.size):
                absorbtion[i]=np.sum(optical/(omega[i]+1j*eta-self.EH))/omega[i]
            spectrum=-absorbtion.imag/np.pi
            spectrum/=np.trapz(spectrum,omega)

            plt.plot(omega,spectrum)
        plt.grid()
    
    def plotExcitonWaveFunction(self,i):
        wave_k,wave_r=self.ExcitonWaveFunction(i)

        plt.figure()
        plt.subplot(211, title="Reciprocal Space")
        plt.scatter(self.K[:,:,0],self.K[:,:,1],c=wave_k)
        plt.xlabel('$\AA^{-1}$');plt.ylabel('$\AA^{-1}$')
        plt.grid()
        plt.axis('equal')
            
        plt.subplot(212, title="Real Space")
        plt.scatter(self.R[:,:,0],self.R[:,:,1],c=wave_r)
        plt.xlabel('$\AA$');plt.ylabel('$\AA$')
        plt.grid()
        plt.axis('equal')
        
        plt.suptitle('Peak: '+str(np.round(self.EH[i],4)))
        