### Performs the MMT inversion to calculate tracer diffusivity maps

import numpy as np
import numpy.linalg as la
import xarray as xr
from xgcm import Grid

# Import fields
root = '/scratch/mp6191/NW2_TracerBackscatter'
exp = '/p03125_2'
res = 0.03125
print(exp)

ds = xr.open_mfdataset(root + exp + '/MMT_fields_*.nc', decode_times = False)
static = xr.open_dataset(root + exp + '/static.nc', decode_times = False)

grid = Grid(ds, coords = {'X': {'center': 'xh', 'outer': 'xq'},
                          'Y': {'center': 'yh', 'outer': 'yq'}})
coarsen_grid = 2
coarsen_scale = int(coarsen_grid / res)

Re = 6.37e6
lat = static['geolat']


# Functions
def TWA(var_h, h, coarsen_scale):
    '''
    TWA of field, var, where the average is a time average (done online) and space average over some coarsening-scale
    '''

    return var_h.mean('time').coarsen(xh = coarsen_scale, yh = coarsen_scale, boundary = 'exact').mean() / h.mean('time').coarsen(xh = coarsen_scale, yh = coarsen_scale, boundary = 'exact').mean()


def Fcol(ds, tracer_name, coarsen_scale):
    '''
    Calculate columns of F matrix: hat(u'' c''), hat(v'' c'')
    '''
    
    # Grid
    grid = Grid(ds, coords = {'X': {'center': 'xh', 'outer': 'xq'},
                              'Y': {'center': 'yh', 'outer': 'yq'}})

    # Fields
    h = ds['h']
    
    hu = ds['uh']
    hu = grid.interp(hu / static.dyCu, axis = 'X')
    hv = ds['vh']
    hv = grid.interp(hv / static.dxCv, axis = 'Y')
    
    hc = ds[tracer_name + 'h']
    
    hcu = ds[tracer_name + '_adx']
    hcu = grid.interp(hcu / static.dyCu, axis = 'X')
    hcv = ds[tracer_name + '_ady']
    hcv = grid.interp(hcv / static.dxCv, axis = 'Y')

    # TWA and TWA-eddy
    c_hat = TWA(hc, h, coarsen_scale)
    u_hat = TWA(hu, h, coarsen_scale)
    uc_hat = TWA(hcu, h, coarsen_scale)
    v_hat = TWA(hv, h, coarsen_scale)
    vc_hat = TWA(hcv, h, coarsen_scale)

    uc_pp_hat = uc_hat - u_hat * c_hat
    vc_pp_hat = vc_hat - v_hat * c_hat

    return uc_pp_hat.values.T, vc_pp_hat.values.T
    

def Gcol(ds, tracer_name, lat, coarsen_scale):
    '''
    Calculate columns of G matrix: d/dx hat(c), d/dy hat(c)
    '''

    # Fields
    h = ds['h']
    hc = ds[tracer_name + 'h']

    # TWA
    c_hat = TWA(hc, h, coarsen_scale)

    # Take derivatives
    c_hat_dx = c_hat.differentiate('xh') * 360 / (2 * np.pi * Re * np.cos(lat.coarsen(xh = coarsen_scale, yh = coarsen_scale, boundary = 'exact').mean() * np.pi / 180))
    c_hat_dy = c_hat.differentiate('yh') * 360 / (2 * np.pi * Re)

    return c_hat_dx.values.T, c_hat_dy.values.T


def Klsq(F, G):
    '''
    Calculates the least squares diffusivity matrix from F, G 
    '''
    # NB: np.moveaxis(x, -2, -1) transposes the last two columns of x, which is the "matrix" part (i.e., first three axes are x, y, z)
    
    F_T = np.moveaxis(F, -2, -1)
    G_T = np.moveaxis(G, -2, -1)
    GG_T = np.matmul(G, G_T)
    GF_T = np.matmul(G, F_T)
    GG_T_inv = la.inv(GG_T)
    K = np.moveaxis(-np.matmul(GG_T_inv, GF_T), -2, -1)
    
    return K


def symm(K):
    '''
    Extracts symmetric part of matrix, K
    '''
    
    K_T = np.moveaxis(K, -2, -1)
    
    return 0.5 * (K + K_T)


def antisymm(K):
    '''
    Extracts antisymmetric part of matrix, K
    '''
    
    K_T = np.moveaxis(K, -2, -1)
    
    return 0.5 * (K - K_T)


# Construct matrices
names = ['tracer01', 'tracer02', 'tracer03', 'tracer04', 'tracer05', 'tracer06', 'tracer07', 'tracer08']

xh = 30
yh = 70
zl = 15
N_trac = len(names)

F = np.zeros((xh, yh, zl, 2, N_trac))
G = np.zeros((xh, yh, zl, 2, N_trac))

for j in range(len(names)):
    F[:, :, :, 0, j], F[:, :, :, 1, j] = Fcol(ds, names[j], coarsen_scale)
    G[:, :, :, 0, j], G[:, :, :, 1, j] = Gcol(ds, names[j], lat, coarsen_scale)

K = Klsq(F, G)
S = symm(K)


# Save full tensor (used for reconstructing flux-gradient relationship)
h = ds['h']
hc = ds['tracer01h']
c_hat = TWA(hc, h, coarsen_scale)
c_hat_dy = (c_hat.differentiate('yh') * 360 / (2 * np.pi * Re)).T

K_full = xr.DataArray(data = K,
                      dims = ['xh', 'yh', 'zl', 'row', 'col'],
                      coords = dict(xh = (['xh'], c_hat_dy.xh.values),
                                    yh = (['yh'], c_hat_dy.yh.values),
                                    zl = (['zl'], c_hat_dy.zl.values),
                                    row = (['row'], [1, 2]),
                                    col = (['col'], [1, 2])))
K_full.name = 'full K'

save_path = root + exp
K_full.to_netcdf(save_path + '/MMT_fullK.nc')


# Calculate major and minor eigenvalues of symmetric part, and store as a 3D arrays that have same shape as, e.g., the coarsened TWA tracer gradient
k1 = np.zeros(c_hat_dy.shape)
k2 = np.zeros(c_hat_dy.shape)

for i in range(k1.shape[0]):
    for j in range(k1.shape[1]):
        for k in range(k1.shape[2]):
            eigvals, eigvecs = la.eig(S[i, j, k, :, :])
        
            indmax = np.abs(eigvals).argmax()
            indmin = np.abs(eigvals).argmin()
        
            k1[i, j, k] = eigvals[indmax]
            k2[i, j, k] = eigvals[indmin]
        
k1 = k1 + 0 * c_hat_dy
k1.name = 'kappa1'

k2 = k2 + 0 * c_hat_dy
k2.name = 'kappa2'

# Save these arrays as netcdf files
save_path = root + exp
k1.to_netcdf(save_path + '/MMT_kappa1.nc')
k2.to_netcdf(save_path + '/MMT_kappa2.nc')

print('done!')
