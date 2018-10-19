import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import numba
import pybinding as pb
from pybinding.repository import group6_tmd
import tbmodels
from scipy.special import struve
from scipy.special import yn
import os

@numba.njit(parallel=True,nogil=True,fastmath=True)
def ExcitonHamiltonian(H,E,shift,D,D_,W,cb,vb,Q):
    N=E.shape[0]
    for kx1 in numba.prange(N):
        for ky1 in range(N):
            for kx2 in range(N):
                for ky2 in range(N):

                    O12=np.dot(D_[(kx1+Q[0])%N,(ky1+Q[1])%N].T,D[(kx2+Q[0])%N,(ky2+Q[1])%N]);
                    O21=np.dot(D_[kx2,ky2].T,D[kx1,ky1]);

                    O11=np.dot(D_[(kx1+Q[0])%N,(ky1+Q[1])%N].T,D[kx1,ky1]);
                    O22=np.dot(D_[kx2,ky2].T,D[(kx2+Q[0])%N,(ky2+Q[1])%N]);

                    for i1,c1 in enumerate(cb):
                        for j1,v1 in enumerate(vb):
                            for i2,c2 in enumerate(cb):
                                for j2,v2 in enumerate(vb):
                                    #direct
                                    H[kx1,ky1,i1,j1,kx2,ky2,i2,j2]-=W[(kx1-kx2)%N,(ky1-ky2)%N]*O12[c1,c2]*O21[v2,v1]
                                    #exchange
                                    H[kx1,ky1,i1,j1,kx2,ky2,i2,j2]+=W[Q[0],Q[1]]*O11[c1,v1]*O22[v2,c2]

            for i1,c1 in enumerate(cb):
                    for j1,v1 in enumerate(vb):
                        #e-h pair
                        H[kx1,ky1,i1,j1,kx1,ky1,i1,j1]+=E[(kx1+Q[0])%N,(ky1+Q[1])%N,c1]-E[kx1,ky1,v1]+shift
    return H

class ExcitonMoS2(object):

    def __init__(self,tbmodel='MoS2-WANN_hr_11.dat'):
        self.py_model = pb.Model(group6_tmd.monolayer_3band("MoS2"), pb.translational_symmetry())
        self.a2,self.a1=self.py_model.lattice.vectors
        self.k1,self.k2=self.py_model.lattice.reciprocal_vectors()
        self.model = tbmodels.Model.from_wannier_files(hr_file='TB-Models/'+str(tbmodel))
        self.norb=self.model.hamilton([0.,0.,0.]).shape[0]

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

    def constuctExcitonHamiltonian(self,epsilon,N,vb,cb,shift=0,Q=[0,0,0],save=False,save_folder='Save'):

        self.epsilon=epsilon #
        self.r0= 3.3875/epsilon #nm
        self.shift=shift # scissor operator
        self.N=N # k-points grid
        self.vb=vb # valence bands
        self.cb=cb # conduction bands
        self.nv=len(self.vb);self.nc=len(self.cb)
        self.NH=self.N**2*self.nc*self.nv
        print('Exciton Hamiltonian size: '+str(self.NH)+'X'+str(self.NH))
        self.Q=Q

        E=np.zeros((N,N,self.norb));K=np.zeros((N,N,3));D=np.zeros((N,N,self.norb,self.norb),dtype=np.complex)
        for i,j in product(range(N),range(N)):
            e,d=np.linalg.eigh(self.model.hamilton([i/N,-j/N,0]))
            E[i,j]=e;D[i,j]=d
            K[i,j]=i*self.k1/N+j*self.k2/N

        R=np.zeros((N,N,3))
        for i,j in product(range(N),range(N)):
            R[i,j]=self.a1*(i-N/2)+self.a2*(j-N/2)

        WR=np.zeros((N,N))
        for i,j in product(range(N),range(N)):
            radius=np.linalg.norm(R[i,j])
            if radius!=0:WR[i,j]=1.44*np.pi/(2*self.epsilon*self.r0)*(struve(0,radius/self.r0)-yn(0,radius/self.r0))
            else:WR[i,j]=0

        WR=np.fft.fftshift(WR);W=np.fft.fft2(WR)/N**2

        H=np.zeros((self.N,self.N,self.nc,self.nv,self.N,self.N,self.nc,self.nv),dtype=np.complex)
        self.H=ExcitonHamiltonian(H,E,shift,D,D.conj(),W,self.cb,self.vb,self.Q)
        print('Construction Done!')
        if save:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            print('Saving Hamiltonian...')
            np.save(save_folder+'/H',self.H)
            print('Saved!')
        self.K=K;self.R=R

    def solveExcitonHamiltonian(self,save=False,save_folder='Save'):
        self.EH,self.DH=np.linalg.eigh(self.H.reshape((self.NH,self.NH)))
        self.DH=self.DH[:,np.argsort(self.EH)]
        self.EH=np.sort(self.EH)
        print('Hamiltonian Solved!')

        if save:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            print('Saving Eigenvalues and Eigenfunctions...')
            np.save(save_folder+'/EH',self.EH)
            np.save(save_folder+'/DH',self.DH)
            print('Saved!')

    def ExcitonWaveFunction(self,i):
        wave_k=np.abs(self.DH[:,i].reshape(self.N**2,self.nc,self.nv))**2
        wave_k=np.sum(wave_k,axis=(1,2))
        wave_k/=np.sum(wave_k);

        wave_k=wave_k.reshape((self.N,self.N))
        wave_r=np.abs(np.fft.fftshift(np.fft.fft2(wave_k)))
        wave_r/=np.sum(wave_r);
        return wave_k,wave_r

    def plotExcitonWaveFunction(self,i):
        wave_k,wave_r=self.ExcitonWaveFunction(i)

        plt.figure()
        plt.subplot(121, title="Reciprocal Space")
        self.py_model.lattice.plot_brillouin_zone()
        plt.contourf(self.K[:,:,0],self.K[:,:,1],wave_k,int(40))
        plt.axis('off')

        plt.subplot(122, title="Real Space")
        plt.contourf(self.R[:,:,0],self.R[:,:,1],wave_r,int(40))
        plt.xlabel('nm');plt.ylabel('nm')
        plt.grid()
        plt.suptitle('Peak: '+str(np.round(self.EH[i],4)))
