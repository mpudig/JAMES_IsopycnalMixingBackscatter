### Calculate mean and variance biases for MMT tracers

import numpy as np
import xarray as xr
import dask
dask.config.set(array__slicing__split_large_chunks = True)

# Import fields
root = '/scratch/mp6191/NW2_TracerBackscatter'

exp = '/p25_noBS_2'
coarsen_scale = 32 // 4

ds = xr.open_mfdataset(root + exp + '/snapshots*.nc', decode_times = False)
static = xr.open_dataset(root + exp + '/static.nc', decode_times = False)

ds_hr = xr.open_mfdataset(root + '/p03125_2' + '/snapshots*.nc', decode_times = False).isel(time = slice(None, 180, 2))

# Functions
def depth_integrate_c(ds, tracer_name):

    c = ds[tracer_name]
    h = ds['h']

    c_int = (h * c).sum('zl') / h.sum('zl')

    return c_int

def mean_tracer_bias(c_int, c_hr_int, area):

    c = c_int.mean('time')
    c_hr = c_hr_int.mean('time')

    ms_bias = ((c - c_hr) ** 2 * area).sum(['xh', 'yh']) / area.sum(['xh', 'yh'])
    
    return np.sqrt(ms_bias.load()).item()


def var_tracer_bias(c_int, c_hr_int, area):

    c = c_int.std('time')
    c_hr = c_hr_int.std('time')
    
    ms_bias = ((c - c_hr) ** 2 * area).sum(['xh', 'yh']) / area.sum(['xh', 'yh'])
    
    return np.sqrt(ms_bias.load()).item()

# Compute
tracer_names = [f'tracer0{i}' for i in range(1, 9)]

mean_biases = xr.Dataset({tracer_names[i]: xr.DataArray(i * 0.0) for i in range(len(tracer_names))})
mean_biases.attrs['exp'] = exp
mean_biases.attrs['type'] = 'mean'

var_biases = xr.Dataset({tracer_names[i]: xr.DataArray(i * 0.0) for i in range(len(tracer_names))})
var_biases.attrs['exp'] = exp
var_biases.attrs['type'] = 'std'

for i in range(len(tracer_names)):
    tracer_name = tracer_names[i]
    
    c_int = depth_integrate_c(ds, tracer_name)
    c_hr_int = depth_integrate_c(ds_hr, tracer_name).chunk({'xh': 192, 'yh': 256}).coarsen(xh = coarsen_scale, yh = coarsen_scale, boundary = 'exact').mean()
    area = static.area_t
    
    mean_biases[tracer_name] = mean_tracer_bias(c_int, c_hr_int, area)
    print(f'{tracer_name} mean done')
    
    var_biases[tracer_name] = var_tracer_bias(c_int, c_hr_int, area)
    print(f'{tracer_name} variance done')

# Save arrays
save_path = '/scratch/mp6191/NW2_TracerBackscatter/MiscFields'
mean_biases.to_netcdf(save_path + exp + '_mean_tracer_biases.nc')
var_biases.to_netcdf(save_path + exp + '_var_tracer_biases.nc')

print('done!')
