import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import tbmodels
from scipy.special import struve,yn
import os
from scipy.linalg import eigh
from ase.units import *
import h5py
from fast_functions import *
from slow_functions import *
from tqdm import tqdm

class ExcitonMoS2(object):

    def __init__(self,N,vb,cb,epsilon,
                 r0=33.875,shift=0.5,spin_orbit=True,cutoff=np.inf):
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
        self.norb=self.model.hamilton([0.,0.,0.]).shape[0]
        self.cutoff=cutoff
        
        self.epsilon=epsilon #
        self.r0= r0/epsilon 
        self.shift=shift # scissor operator
        self.N=N # k-points grid
        self.vb=vb # valence bands
        self.cb=cb # conduction bands
        self.nv=len(self.vb);self.nc=len(self.cb)
        
        

        self.E=np.zeros((N,N,self.norb));self.K=np.zeros((N,N,3));
        self.D=np.zeros((N,N,self.norb,self.norb),dtype=np.complex)
        self.H=np.zeros((N,N,self.norb,self.norb),dtype=np.complex)
        for i,j in product(range(N),range(N)):
            self.H[i,j]=self.model.hamilton([i/N,j/N,0])
            e,d=np.linalg.eigh(self.H[i,j])
            self.E[i,j]=e;self.D[i,j]=d
            self.K[i,j]=i*self.k1/N+j*self.k2/N

        R=np.zeros((N,N,3))
        for i,j in product(range(N),range(N)):
            R[i,j]=self.a1*(i-N/2)+self.a2*(j-N/2)
        self.R=R
        WR=np.zeros((N,N));VR=np.zeros((N,N))
        for i,j in product(range(N),range(N)):
            radius=np.linalg.norm(R[i,j])
            if radius!=0:
                WR[i,j]=Hartree*Bohr*np.pi/(2*self.epsilon*self.r0)*(struve(0,radius/self.r0)-yn(0,radius/self.r0))
                VR[i,j]=Hartree*Bohr/radius
            else:
                WR[i,j]=0
                VR[i,j]=0
            

        WR=np.fft.fftshift(WR);self.W=np.fft.fft2(WR)/N**2
        VR=np.fft.fftshift(VR);self.V=np.fft.fft2(VR)/N**2
        
        self.gap=self.E[:,:,cb[0]]-self.E[:,:,vb[-1]]+shift
        self.Egap=np.min(self.gap)
        self.weight=np.ones((self.N,self.N))
        
        self.weight[self.gap>(self.Egap+self.cutoff)]=0
        self.indexes=[]
        for kx,ky in product(range(self.N),range(self.N)):
            if self.weight[kx,ky]==1:
                for i,j in product(self.cb,self.vb):
                    self.indexes.append((kx,ky,i,j))
        self.NH=len(self.indexes)
#         print('Exciton Hamiltonian size: '+str(self.NH)+' K-space size: '+str(int(np.sum(self.weight))))
        
  
        R=np.fft.fftshift(self.R,axes=(0,1))
        HR=np.fft.ifft2(self.H,axes=(0,1))
        dx=np.fft.fft2(1j*R[:,:,0,None,None]*HR,axes=(0,1))
        dy=np.fft.fft2(1j*R[:,:,1,None,None]*HR,axes=(0,1))
        for i,j in product(range(N),range(N)):
            dx[i,j]=np.linalg.multi_dot([self.D[i,j].T.conj(),dx[i,j],self.D[i,j]])
            dy[i,j]=np.linalg.multi_dot([self.D[i,j].T.conj(),dy[i,j],self.D[i,j]])
        self.dx=dx;self.dy=dy
        self.d=dx
        
    
    def constructTrionBasis(self,trion_q=[(1./3.),(1./3.),0]):
        self.trion_q=np.array([q*self.N for q in trion_q],dtype=int)
        self.trion_spectrum_shift=self.E[self.trion_q[0],self.trion_q[1],self.cb[0]]
        self.trion_indexes=[]
        
        self.conduction_states=[(kx,ky,c) for kx,ky,c in product(range(self.N),range(self.N),self.cb)]
        self.valence_states=[(kx,ky,v) for kx,ky,v in product(range(self.N),range(self.N),self.vb)]
        
        self.nc=len(self.conduction_states)
        self.nv=len(self.valence_states)
        
        self.trion_indexes=FastTrionBasis(self.conduction_states,self.valence_states, self.trion_q,self.weight)              
        self.NT=len(self.trion_indexes)
        
    def constructTrionHamiltonian(self):
        HT=np.zeros((self.NT,self.NT),dtype=complex)
        self.HT=FastTrionHamiltonian(HT,self.E,self.D,self.W,self.V,
                                     self.conduction_states,self.valence_states,
                                     self.trion_indexes,self.shift)

    def constuctExcitonHamiltonian(self,Q=[0,0,0]):
        self.Q=np.array([q*self.N for q in Q],dtype=int)
        HH=np.empty((self.NH,self.NH),dtype=np.complex)
        self.HH=FastExcitonHamiltonian(HH,self.E,self.shift,self.D,self.W,self.V,self.indexes,self.Q)
    
    def calculateAbsobtionSpectrumTrion(self,eta=0.03,omega_max=5,omega_n=50000,n_iter=300):
        omega=np.linspace(0,omega_max,omega_n+1)+1j*eta
        omega=np.delete(omega,0)
        P=np.zeros(self.NT,dtype=complex)
        for i in range(self.NT):
            i1,i2,j=self.trion_indexes[i]
            x1,y1,c1=self.conduction_states[i1]
            x2,y2,c2=self.conduction_states[i2]
            xv,yv,v=self.valence_states[j]
            if x1==self.trion_q[0] and y1==self.trion_q[1]:
                if x2==xv and y2==yv:
                    P[i]+=(self.d[x2,y2,c2,v])
            if x2==self.trion_q[0] and y2==self.trion_q[1]:
                if x1==xv and y1==yv:
                    P[i]-=(self.d[x1,y1,c1,v])
                
        a,b=lanczos(self.HT,P,n_iter)
        eps=np.zeros(omega.size,dtype=complex)
        for i in range(1,n_iter):
            eps=b[-i]**2/(omega+1j*eta-a[-i]-eps)
        eps=1/(omega+1j*eta-a[0]-eps)
        self.trion_eps=eps
        self.trion_omega=omega
        
        
        
    def calculateAbsobtionSpectrum(self,eta=0.03,omega_max=5,omega_n=50000,n_iter=300):     
        omega=np.linspace(0,omega_max,omega_n+1)+1j*eta
        omega=np.delete(omega,0) 
        
        P=np.array([self.d[indx] for indx in self.indexes])
        a,b=lanczos(self.HH,P,n_iter)
            
        eps=np.zeros(omega.size,dtype=complex)
        for i in range(1,n_iter):
            eps=b[-i]**2/(omega+1j*eta-a[-i]-eps)
        eps=1/(omega+1j*eta-a[0]-eps)
        self.eps=eps
        self.omega=omega
        
    def plotAbsobtionSpectrum(self):
        plt.figure()
        if hasattr(self, 'omega') and hasattr(self, 'eps'):
            plt.plot(self.omega.real,-self.eps.imag,label='Exciton')
            
        if hasattr(self, 'trion_omega') and hasattr(self, 'trion_eps'):
            
            plt.plot(self.trion_omega.real-self.trion_spectrum_shift,-self.trion_eps.imag,label='Trion')
        plt.grid()
        plt.legend()
        plt.xlim([0,None])
        
        
    def plotBandStructure(self,N1=20,emax=5,emin=-8,E_Fermi=0):
        gamma=np.array([0.,0.,0.])
        M=np.array([0,0.5,0.])
        K=np.array([1./3.,1./3.,0.])
        kpoints=[];N1=N1;N2=int(N1/2);N3=int(np.sqrt(N1**2+N2**2))
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