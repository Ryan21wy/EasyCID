import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve


def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=30):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_,porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print ('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z


# from spc.spcio import readSPC
# import os
# from matplotlib import pyplot as plt
#
#
# dir = r'C:\Users\Sunshine\Documents\WeChat Files\wxid_ab9mcmiea04722\FileStorage\File\2021-11\python\methylephedrine'
# dir_list = os.listdir(dir)
# for file in dir_list:
#     axis, x, y = readSPC(os.path.join(dir, file))
#     raw_axis = x.reshape(-1)[6:-125]
#     raw_spectrum = y.reshape(-1)[6:-125]
#     bg = airPLS(raw_spectrum, 10, 1, 100)
#     new_spectrum = raw_spectrum - bg
#     # plt.plot(raw_axis, new_spectrum, label='BR')
#     new_spectrum = WhittakerSmooth(new_spectrum, np.ones(new_spectrum.shape[0]), 2)
#     # plt.plot(raw_axis, new_spectrum, label='SS')
#     plt.plot(raw_axis, raw_spectrum)
#     # plt.plot(raw_axis, bg, label='BG')
#     # plt.plot(raw_axis, new_spectrum, label='BR')
#     plt.title(str(file.split('.')[0]))
#     # plt.legend()
#     plt.show()