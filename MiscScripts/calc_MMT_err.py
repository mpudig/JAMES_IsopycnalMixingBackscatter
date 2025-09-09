### Performs the MMT inversion standard error calculation

import numpy as np
import numpy.linalg as la
import xarray as xr
from xgcm import Grid

# Import fields
root = '/scratch/mp6191/NW2_TracerBackscatter'
exp = '/p25_noBS_2'
res = 0.25
print(exp)

ds = xr.open_mfdataset(root + exp + '/MMT_fields*.nc', decode_times = False)
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


# Construct matrices for 2 yr time scale
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

# To have a nice field to save fields like
h = ds.isel(time = slice(0, 1))['h']
hc = ds.isel(time = slice(0, 1))['tracer01h']
c_hat = TWA(hc, h, coarsen_scale)
c_hat_dy = (c_hat.differentiate('yh') * 360 / (2 * np.pi * Re)).T

# Compute standard errors
def compute_std_err_kappa(F, G):
    # Do pseudoinversion
    A = np.kron(-np.moveaxis(G, -2, -1), np.identity(2))
    Adag = la.pinv(A)
    shape = F.shape[:-2] + (F.shape[-2] * F.shape[-1],)
    f = F.reshape(shape, order = 'F')
    khat = np.matmul(Adag, f[..., None])[..., 0]
    
    # Calculate residuals, covariances and standard errors of K
    r = f - np.matmul(A, khat[..., None])[..., 0]
    stat_dof = 2 * 8 - 4
    s_sq = la.norm(r, axis = -1) ** 2 / stat_dof
    
    cov_K = s_sq[..., None, None] * la.inv(np.matmul(np.moveaxis(A, -2, -1), A))
    se_K = np.sqrt(cov_K)
    
    # Function that maps K to kappas
    a = khat[:, :, :, 0]
    b = khat[:, :, :, 1]
    c = khat[:, :, :, 2]
    d = khat[:, :, :, 3]
    
    m = (a + d) / 2
    n = (a - d) / 2
    s = (b + c) / 2
    p = np.sqrt(n ** 2 + s ** 2)
    
    # Calculate Jacobian of this function
    row1 = np.stack([1 + n/p, s/p, s/p, 1 - n/p], axis = -1)
    row2 = np.stack([1 - n/p, -s/p, -s/p, 1 + n/p], axis = -1)
    J = 0.5 * np.stack([row1, row2], axis = -2)
    
    # Calculate covariance and standard errors of kappas
    cov_kappa = np.matmul(J, np.matmul(cov_K, np.moveaxis(J, -2, -1)))
    se_kappa = np.diagonal(np.sqrt(cov_kappa), axis1 = -2, axis2 = -1)
    se_kappa1 = se_kappa[:, :, :, 0]
    se_kappa2 = se_kappa[:, :, :, 1]

    return se_kappa1, se_kappa2

k1_err, k2_err = compute_std_err_kappa(F, G)
k1_err = k1_err + 0 * c_hat_dy
k2_err = k2_err + 0 * c_hat_dy

# Rename
#k1_err.name = 'kappa1'
#k2_err.name = 'kappa2'

# Save these arrays as netcdf files
save_path = root + exp
k1_err.to_netcdf(save_path + '/MMT_kappa1_err.nc')
k2_err.to_netcdf(save_path + '/MMT_kappa2_err.nc')

print('done!')
