import numpy as np
import numba

@numba.njit()
def FastTrionBasis(cs,vs,q,weight):
    nc=len(cs);nv=len(vs)
    trion_indexes=[]
    N=weight.shape[0]
    for i1 in range(nc):
        kx1,ky1,c1=cs[i1]
        if weight[kx1,ky1]==1:
            for i2 in range(i1+1,nc):
                kx2,ky2,c2=cs[i2]
                if weight[kx2,ky2]==1:
                    for j in range(nv):
                        kxv,kyv,v=vs[j]
                        if weight[kxv,kyv]==1:
                            if (kx1+kx2-kxv)%N==q[0] and (ky1+ky2-kyv)%N==q[1]:
                                trion_indexes.append((i1,i2,j))
    return trion_indexes
        


@numba.njit(parallel=True,nogil=True,fastmath=True)
def FastExcitonHamiltonian(H,E,shift,D,W,V,indexes,q):
    NH=H.shape[0];N=E.shape[0]
    for i1 in numba.prange(NH):
        x1,y1,c1,v1=indexes[i1]
        H[i1,i1]=E[(x1+q[0])%N,(y1+q[1])%N,c1]-E[x1,y1,v1]+shift
        
        overlap_c=np.dot(D[(x1+q[0])%N,(y1+q[1])%N,:,c1].T.conjugate(),D[(x1+q[0])%N,(y1+q[1])%N,:,c1]);
        overlap_v=np.dot(D[x1,y1,:,v1].T.conjugate(),D[x1,y1,:,v1]);
        H[i1,i1]-=W[(x1-x1)%N,(y1-y1)%N]*overlap_c*overlap_v
        
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
def FastTrionHamiltonian(H,E,D,W,V,cs,vs,indexes,shift): 
    NH=len(indexes);N=E.shape[0]
    for indx in numba.prange(NH):   
        
        i1,i2,j=indexes[indx]
        x1,y1,c1=cs[i1]
        x2,y2,c2=cs[i2]
        xv,yv,v=vs[j]
        
        H[indx,indx]=E[x1,y1,c1]+E[x2,y2,c2]-E[xv,yv,v]+shift    
        for indx_ in range(indx,NH):   
            
            i1_,i2_,j_=indexes[indx_]
            x1_,y1_,c1_=cs[i1_]
            x2_,y2_,c2_=cs[i2_]
            xv_,yv_,v_=vs[j_]
            
            element=0
            if xv==xv_ and yv==yv_ and v==v_:
                
                o_1=np.dot(D[x1,y1,:,c1].T.conjugate(),D[x1_,y1_,:,c1_])
                o_2=np.dot(D[x2,y2,:,c2].T.conjugate(),D[x2_,y2_,:,c2_])
                element+=W[(x1-x1_)%N,(y1-y1_)%N]*o_1*o_2
                
                o_1=np.dot(D[x1,y1,:,c1].T.conjugate(),D[x2_,y2_,:,c2_])
                o_2=np.dot(D[x2,y2,:,c2].T.conjugate(),D[x1_,y1_,:,c1_])
                element-=W[(x1-x2_)%N,(y1-y2_)%N]*o_1*o_2
                
            if x2==x2_ and y2==y2_ and c2==c2_:
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[xv,yv,:,v])
                o_2=np.dot(D[x1,y1,:,c1].T.conjugate(),D[x1_,y1_,:,c1_])
                element-=W[(xv_-xv)%N,(yv_-yv)%N]*o_1*o_2
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[x1_,y1_,:,c1_])
                o_2=np.dot(D[x1,y1,:,c1].T.conjugate(),D[xv,yv,:,v])
                element+=V[(x1-xv)%N,(y1-yv)%N]*o_1*o_2
                
            if x1==x1_ and y1==y1_ and c1==c1_:
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[xv,yv,:,v])
                o_2=np.dot(D[x2,y2,:,c2].T.conjugate(),D[x2_,y2_,:,c2_])
                element-=W[(xv_-xv)%N,(yv_-yv)%N]*o_1*o_2
                
                o_1=np.dot(D[xv_,yv_,:,v_].T.conjugate(),D[x2_,y2_,:,c2_])
                o_2=np.dot(D[x2,y2,:,c2].T.conjugate(),D[xv,yv,:,v])
                element+=V[(x2-xv)%N,(y2-yv)%N]*o_1*o_2
                
            if indx==indx_:
                H[indx,indx]+=element
            else:
                H[indx,indx_]=element
                H[indx_,indx]=element.conjugate()

    return H
