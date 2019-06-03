import numpy as np

def lanczos(H,P,n_iter):
    a=[];b=[];vec=[]
    P/=np.linalg.norm(P)
    vec.append(P)
    a.append(vec[-1].conj()@H@vec[-1])
    vec.append(H@vec[-1]-a[-1]*vec[-1])
    vec[-1]/=np.linalg.norm(vec[-1])
    b.append(vec[-2].conj()@H@vec[-1])

    for n in range(1,n_iter):
        a.append(vec[-1].conj()@H@vec[-1])
        b.append(vec[-2].conj()@H@vec[-1])
        vec.append(H@vec[-1]-a[-1]*vec[-1]-b[-1]*vec[-2])
        vec[-1]/=np.abs(np.linalg.norm(vec[-1]))
        del vec[0]
        
    return a,b