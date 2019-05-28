import numpy as np
import numba

@numba.njit(parallel=True,nogil=True,fastmath=True)
def FastExcitonHamiltonianOptical(H,E,shift,D,W,indexes):
    NH=H.shape[0];N=E.shape[0]
    for i1 in numba.prange(NH):
        x1,y1,c1,v1=indexes[i1]
        H[i1,i1]=E[x1,y1,c1]-E[x1,y1,v1]+shift-W[0,0]
        for i2 in range(i1+1,NH):
            x2,y2,c2,v2=indexes[i2]
            overlap_c=np.sum(D[x1,y1,:,c1].conjugate()*D[x2,y2,:,c2]);
            overlap_v=np.sum(D[x2,y2,:,v2].conjugate()*D[x1,y1,:,v1]);
            H[i1,i2]=-W[(x1-x2)%N,(y1-y2)%N]*overlap_c*overlap_v   
            H[i2,i1]=H[i1,i2].conjugate()
    return H

@numba.njit(parallel=True,nogil=True,fastmath=True)
def FastExcitonHamiltonian(H,E,shift,D,W,V,indexes,q):
    NH=H.shape[0];N=E.shape[0]
    for i1 in numba.prange(NH):
        x1,y1,c1,v1=indexes[i1]
        H[i1,i1]=E[(x1+q[0])%N,(y1+q[1])%N,c1]-E[x1,y1,v1]+shift-W[0,0]
        
        overlap_c=np.dot(D[(x1+q[0])%N,(y1+q[1])%N,:,c1].T.conjugate(),D[x1,y1,:,v1]);
        overlap_v=np.dot(D[x1,y1,:,v1].T.conjugate(),D[(x1+q[0])%N,(y1+q[1])%N,:,c1]);
        H[i1,i1]+=V[q[0],q[1]]*overlap_c*overlap_v
        
        for i2 in range(i1+1,NH):
            x2,y2,c2,v2=indexes[i2]
            overlap_c=np.dot(D[(x1+q[0])%N,(y1+q[1])%N,:,c1].T.conjugate(),D[(x2+q[0])%N,(y2+q[1])%N,:,c2]);
            overlap_v=np.dot(D[x2,y2,:,v2].T.conjugate(),D[x1,y1,:,v1]);
            H[i1,i2]=-W[(x1-x2)%N,(y1-y2)%N]*overlap_c*overlap_v

            overlap_c=np.dot(D[(x1+q[0])%N,(y1+q[1])%N,:,c1].T.conjugate(),D[x1,y1,:,v1]);
            overlap_v=np.dot(D[x2,y2,:,v2].T.conjugate(),D[(x2+q[0])%N,(y2+q[1])%N,:,c2]);
            H[i1,i2]+=V[q[0],q[1]]*overlap_c*overlap_v
            
            H[i2,i1]=H[i1,i2].conjugate()
    return H


@numba.njit(parallel=True,nogil=True,fastmath=True)
def FastTrionHamiltonian(H,E,D,W,V,indexes,shift): 
    NH=H.shape[0];N=E.shape[0]
    for indx1 in numba.prange(NH):   
        xc1,yc1,xc2,yc2,xv,yv,c1,c2,v=indexes[indx1]
        H[indx1,indx1]+=E[xc1,yc1,c1]+E[xc2,yc2,c2]-E[xv,yv,v]+shift  
        for indx2 in range(NH):   
            xc1_,yc1_,xc2_,yc2_,xv_,yv_,c1_,c2_,v_=indexes[indx2]
            if xv==xv_ and yv==yv_ and v==v_:
                
                o_1=np.dot(D[xc1,yc1,:,c1].T.conjugate(),D[xc1_,yc1_,:,c1_])
                o_2=np.dot(D[xc2,yc2,:,c2].T.conjugate(),D[xc2_,yc2_,:,c2_])
                H[indx1,indx2]+=W[(xc1-xc1_)%N,(yc1-yc1_)%N]*o_1*o_2
                
                o_1=np.dot(D[xc1,yc1,:,c1].T.conjugate(),D[xc2_,yc2_,:,c2_])
                o_2=np.dot(D[xc2,yc2,:,c2].T.conjugate(),D[xc1_,yc1_,:,c1_])
                H[indx1,indx2]-=W[(xc1-xc2_)%N,(yc1-yc2_)%N]*o_1*o_2
                
            if xc2==xc2_ and yc2==yc2_ and c2==c2_:
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[xv,yv,:,v])
                o_2=np.dot(D[xc1,yc1,:,c1].T.conjugate(),D[xc1_,yc1_,:,c1_])
                H[indx1,indx2]-=W[(xv_-xv)%N,(yv_-yv)%N]*o_1*o_2
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[xc1_,yc1_,:,c1_])
                o_2=np.dot(D[xc1,yc1,:,c1].T.conjugate(),D[xv,yv,:,v])
                H[indx1,indx2]+=V[(xc1-xv)%N,(yc1-yv)%N]*o_1*o_2
                
            if xc1==xc1_ and yc1==yc1_ and c1==c1_:
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[xv,yv,:,v])
                o_2=np.dot(D[xc2,yc2,:,c2].T.conjugate(),D[xc2_,yc2_,:,c2_])
                H[indx1,indx2]-=W[(xv_-xv)%N,(yv_-yv)%N]*o_1*o_2
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[xc2_,yc2_,:,c2_])
                o_2=np.dot(D[xc2,yc2,:,c2].T.conjugate(),D[xv,yv,:,v])
                H[indx1,indx2]+=V[(xc2-xv)%N,(yc2-yv)%N]*o_1*o_2
                
#             if indx1!=indx2:
#                 H[indx2,indx1]=H[indx1,indx2].conjugate()
    return H


