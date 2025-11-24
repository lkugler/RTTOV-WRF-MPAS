#!/glade/u/home/lkugler/.local/share/mamba/envs/lkugler-Py310/bin/python
import os, time, warnings
import sys
import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
from scipy.spatial import cKDTree
#from pysolar.solar import get_altitude, get_azimuth
import pvlib  # for solar position calculations
# https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov12/ir_srf/rtcoef_msg_4_seviri_srf.html


def get_solarangles(times, latitude, longitude, altitude, method='nrel_numpy'):
    """
    Calculates the solar zenith angle (SZA), and azimuth angle for given locations and times.

    Parameters
    ----------
    latitude : float or array-like
        Observer's latitude(s) in degrees.
    longitude : float or array-like
        Observer's longitude(s) in degrees.
    times : pandas.DatetimeIndex
        Times for which to calculate the solar position. Must be timezone-localized 
        (e.g., to UTC or a local timezone).
    altitude : float or array-like, optional
        Observer's altitude(s) in meters. Default is 0.
    method : str, optional
        Solar position algorithm to use. 'nrel_numpy' is the default and a good 
        balance of speed and accuracy.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The apparent solar zenith angle and azimuth angle in degrees.
    """
    assert len(times) == len(latitude) == len(longitude), "Length of times, latitude, and longitude must be the same."
    
    times = pd.DatetimeIndex(times).tz_localize('UTC')
    sun_zen = np.zeros((len(times)))
    sun_azi = np.zeros((len(times)))
    
    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        method=method
    )
    sun_zen = solpos['zenith']
    sun_azi = solpos['azimuth']
    # # 'apparent_zenith' includes corrections for atmospheric refraction
    return sun_zen, sun_azi

def expand(nprofiles, input_singlecolumn):
    return np.ones((nprofiles, 1))*input_singlecolumn

def theta_to_tk(theta, rho, qv):
    """Convert dry potential temperature to sensible temperature
    Taken from model_mod.f90 in DART code

    Args:
        theta (xr.DataArray): dry potential temperature in K
        rho (xr.DataArray): dry air density [kg/m3]
        qv (xr.DataArray): water vapor mixing ratio [kg/kg]

    Returns:
        xr.DataArray: sensible temperature in K
    """
    p0 = 100000.0  # Reference pressure [Pa]
    rgas = 287.0  # Constant: Gas constant for dry air [J kg-1 K-1]
    rv = 461.6    # Constant: Gas constant for water vapor [J kg-1 K-1]
    cp = 7.0*rgas/2.0  # = 1004.5 Constant: Specific heat of dry air at constant pressure [J kg-1 K-1] 
    rcv = rgas/(cp-rgas)
    rvord = rv/rgas    
    
    # force qv to be non-negative
    qv = np.maximum(qv, 0.)
    theta_m = (1. + rvord * qv)*theta

    exner = ( (rgas/p0) * (rho*theta_m) )**rcv
    tk = theta_m * exner
    return tk

def lonlat_to_xyz(lon, lat, input_degrees=False):
    """Converts spherical coordinates to Cartesian (to support cKDTree mapping).
    
    Args:
        lon (np.array): longitudes in radians or degrees
        lat (np.array): latitudes in radians or degrees
        input_degrees (bool): if True, input lon/lat are in degrees
    
    Returns:
        np.array: shape (npoints, 3) with x,y,z coordinates    
    """
    if input_degrees:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])

def find_closest_cell_center(query_coords, grid_file):
    """Map all observation points to the nearest MPAS cell using cKDTree
    
    For each observed location, we select the closest model grid point
    
    Args:
        query_coords (tuple): (lon, lat) of query points as 1D arrays
        grid_file (str): path to MPAS model grid netcdf file

    Returns:
    
        np.array: cell indices closest to each query point (1-based indexing)
            np.nan if no cell within max_distance_km
            shape (nprofiles,) = length of query points
    """
    with xr.open_dataset(grid_file, engine='netcdf4') as ds_model_grid:
        lonC_rad = ds_model_grid['lonCell'].values.ravel()
        latC_rad = ds_model_grid['latCell'].values.ravel()
    tree = cKDTree(lonlat_to_xyz(lonC_rad, latC_rad, input_degrees=False))

    query_lon = query_coords[0]
    query_lat = query_coords[1]
    dist, idx = tree.query(lonlat_to_xyz(query_lon, query_lat, input_degrees=True), k=1)
    
    # convert distance from radians to kilometers
    earth_radius_km = 6371.0
    dist_km_approx = dist * earth_radius_km
        
    assert len(idx) == len(query_lon), "Length of output indices must match number of query points."
    return dist_km_approx, idx

class MPAS_Data:
    """Class to hold model data for RTTOV processing
    """
    def __init__(self, ds, config): 
        self.grid_file = config['model_grid']['file']
        self.nlevels = ds.sizes['nVertLevels']
        self.nCells = ds.sizes['nCells']
        self.invalid_mask = None  # mask for invalid model points after interpolation to obs locations
        
        self.vars_3d = dict(
            p = (ds['pressure_base'] + ds['pressure_p'])/100,  # pressure  hPa
            theta = ds['theta'],  # dry potential temperature  K
            rho =  ds['rho'],     # dry air density  kg m-3
            vapor = ds['qv'],     #  Water vapor mixing ratio  kg kg-1
            cloud_water = ds['qc'],    # Cloud liquid water mixing ratio  kg kg-1
            cloud_ice = ds['qi'],      # Cloud ice mixing ratio  kg kg-1
            cloud_snow = ds['qs'],     # Cloud snow mixing ratio  kg kg-1
            re_cloud = ds['re_cloud']*1e6,  # cloud droplet effective radius in microns
            re_ice = ds['re_ice']*1e6,      # ice crystal effective radius in microns
            re_snow = ds['re_snow']*1e6,    # snow effective radius in microns
            cldfrac = ds['cldfrac'],        # cloud fraction 0-1
        )
        
        # variables without vertical dimension
        self.vars_2d = dict(
            emissivity = ds['sfc_emiss'],  # surface emissivity
            terrain = ds['ter'],  # terrain height in m
            skintemp = ds['skintemp'],  # skin temperature in K
            u10 = ds['u10'], 
            v10 = ds['v10'],  # 10m wind components in m/s
            psfc = ds['surface_pressure']/100.,  # surface pressure in hPa
        )
        
        # squeeze time axis if len(time)==1
        if ds.sizes['Time'] > 1:
            raise NotImplementedError('MPAS_Data currently only supports single time slices.')
        for varname, var in self.vars_3d.items():
            if 'Time' in list(var.dims):
                self.vars_3d[varname] = var.isel(Time=0)
        for varname, var in self.vars_2d.items():
            if 'Time' in list(var.dims):
                self.vars_2d[varname] = var.isel(Time=0)
                
        # force vapor into limits
        self.vars_3d['vapor'] = np.maximum(self.vars_3d['vapor'], 1e-10)


    def interp_to_obs(self, obs_coords_deg, method='nearest', max_distance_km=15):
        """Interpolate model data to observation locations
        
        Args:
            obs_coords_deg (tuple): (lon, lat) of observation points as 1D arrays
            method (str): currently only 'nearest' is implemented
            max_distance_km (float): maximum distance to consider a model cell valid
        """
        t = time.time()
        if method == 'nearest':
            dist_km_approx, cellID_for_obs = find_closest_cell_center(obs_coords_deg, 
                                                                      self.grid_file)
            # select model variables at those cell IDs
            for varname, var in self.vars_3d.items():
                self.vars_3d[varname] = var.values[cellID_for_obs, :]
            for varname, var in self.vars_2d.items():
                self.vars_2d[varname] = var.values[cellID_for_obs]

            # set model variables to nan where distance > max_distance_km
            self.invalid_mask = (dist_km_approx > max_distance_km)
            # set output to nan later at these points
        else:
            raise NotImplementedError('Only nearest neighbor interpolation is implemented.')
        
        t = time.time()-t
        print('interp_to_obs() took', int(t*10)/10., 's')
        

    def get_sensible_temperature(self):
        """Compute sensible temperature from dry potential temperature

        Returns:
            np.array: sensible temperature in K
        """
        return theta_to_tk(self.vars_3d['theta'], self.vars_3d['rho'], self.vars_3d['vapor'])

    def flip_vertical_dimension(self):
        """Flip vertical dimension from ground-to-TOA to TOA-to-ground
        
        TOA-to-ground is required by RTTOV
        """
        for varname, arr in self.vars_3d.items():
            self.vars_3d[varname] = np.flip(arr, axis=1)

    def reshape_2d_vars_for_rttov(self):
        """Reshape 2d variables to (-1,) for RTTOV processing
        """
        for varname, arr in self.vars_2d.items():
            self.vars_2d[varname] = arr.values.reshape((-1,))
            
class ObsMetadata:
    """Class to hold observation data for RTTOV processing
    """
    def __init__(self, filename, config, default_time=None):
        self.lat = None
        self.lon = None
        self.satzen = None
        self.satazi = None
        self.obs_time = None

        if filename:
            try:
                dsobs = xr.open_dataset(filename, engine='netcdf4')
            except:
                # set all to default values
                self.set_satazi_from_default(config)
                self.set_satzen_from_default(config)
                self.set_obs_time_from_default(default_time)
            else:
                # dsobs is successfully opened
                try:
                    self.lat = dsobs[config['observations']['latitude']['variable_name']].values.ravel()
                except KeyError:
                    warnings.warn(f"Latitude variable '{config['observations']['latitude']['variable_name']}' not found in observation file, get output on model grid.")
                try:
                    lon = dsobs[config['observations']['longitude']['variable_name']].values.ravel()
                    if config['observations']['longitude'].get('range_0_360', False):
                        # longitudes are in range 0 to 360 degrees, convert to -180 to 180
                        lon = ((lon + 180.) % 360.) - 180.
                    self.lon = lon
                except KeyError:
                    warnings.warn(f"Longitude variable '{config['observations']['longitude']['variable_name']}' not found in observation file, get output on model grid.")

                # get satellite angles, use default values if not present
                try:
                    self.satzen = dsobs[config['observations']['sat_zenith_angle']['variable_name']].values.ravel()
                except KeyError:
                    warnings.warn(f"Satellite zenith angle variable '{config['observations']['sat_zenith_angle']['variable_name']}' not found in observation file, using default value.")
                    self.set_satzen_from_default(config)
                try:
                    self.satazi = dsobs[config['observations']['sat_azimuth_angle']['variable_name']].values.ravel()
                except KeyError:
                    warnings.warn(f"Satellite azimuth angle variable '{config['observations']['sat_azimuth_angle']['variable_name']}' not found in observation file, using default value.")
                    self.set_satazi_from_default(config)
        
                # get observation time, use default value if not present in obs file
                try:
                    self.obs_time = dsobs[config['observations']['observation_time']['variable_name']].values.ravel()
                except KeyError:
                    warnings.warn(f"Observation time variable '{config['observations']['observation_time']['variable_name']}' not found in observation file, using default value.")
                    self.set_obs_time_from_default(default_time)
                dsobs.close()
        else:
            self.set_obs_time_from_default(default_time)
            self.set_satazi_from_default(config)
            self.set_satzen_from_default(config)

    def set_satzen_from_default(self, config):
        self.satzen = np.full(1, config['observations']['sat_zenith_angle']['default_value'])
        
    def set_satazi_from_default(self, config):
        self.satazi = np.full(1, config['observations']['sat_azimuth_angle']['default_value'])

    def set_obs_time_from_default(self, default_time):
        if default_time is None:
            raise ValueError(f"Provide --obs_time.")
        else:
            self.obs_time = default_time
            
            
class ModelGrid:
    """Class to hold model grid data for RTTOV processing
    """
    def __init__(self, grid_file): 
        with xr.open_dataset(grid_file, engine='netcdf4') as ds_model_grid:
            self.lonC_rad = ds_model_grid['lonCell'].values.ravel()
            self.latC_rad = ds_model_grid['latCell'].values.ravel()
            self.nCells = ds_model_grid.sizes['nCells']
            
            # MPAS longitudes are in range 0 to 2pi radian
            lonC_deg = np.rad2deg(self.lonC_rad)
            self.lonC_deg = ((lonC_deg + 180.) % 360.) - 180.
            self.latC_deg = np.rad2deg(self.latC_rad)
    
def call_pyrttov(ds, obs_md, instrument, irAtlas, config, default_time=None):
    """Run RTTOV, return xarray Dataset of reflectance or brightness temperature
    
    We read observation locations, times, and satellite angles from netcdf.

    Args:
        ds (xr.Dataset): instance returned by xr.open_dataset, select one time
        instrument (pyrttov.Instrument): instrument instance
        irAtlas (pyrttov.Atlas): IR atlas instance
        config (dict): configuration dictionary
        obs_md (ObsMetadata, optional): observation metadata instance
        default_time (dt.datetime, optional): default observation time if not in obs_md

    Returns:
        xr.Dataset: with RTTOV output variables
    """
    model = MPAS_Data(ds, config)
    modelgrid = ModelGrid(config['model_grid']['file'])
    
    need_to_interpolate = (obs_md.lat is not None) and (obs_md.lon is not None)
    lon = obs_md.lon if obs_md.lon is not None else modelgrid.lonC_deg
    lat = obs_md.lat if obs_md.lat is not None else modelgrid.latC_deg
    nobs = len(lon)
    
    # make sure these are same len as lon/lat
    satzen = obs_md.satzen if len(obs_md.satzen) == nobs else np.full(nobs, obs_md.satzen[0])
    satazi = obs_md.satazi if len(obs_md.satazi) == nobs else np.full(nobs, obs_md.satazi[0])

    if obs_md.obs_time is None:
        obs_time = default_time  # take default from command line
    else:
        obs_time = obs_md.obs_time  # time info from obs file
    assert obs_time is not None, "Either obs_md.obs_time or default_time must be provided."
    obs_time = np.full(nobs, obs_time.astype('datetime64[s]'))
    
    nprofiles = len(obs_time)
    assert len(lon) == nprofiles, "Length of lon must match number of profiles."
    assert len(lat) == nprofiles, "Length of lat must match number of profiles."
    assert len(satzen) == nprofiles, "Length of satzen must match number of profiles."
    assert len(satazi) == nprofiles, "Length of satazi must match number of profiles."
    
    if need_to_interpolate:  # interpolate model to observation locations
        obs_coords_deg = (lon, lat)
        model.interp_to_obs(obs_coords_deg, 
                            method=config['observations']['interp_model']['method'],
                            max_distance_km=config['observations']['interp_model']['max_distance_km'])

        
    # RTTOV convention
    model.flip_vertical_dimension()  # RTTOV requires TOA-to-ground ordering
    model.reshape_2d_vars_for_rttov()  # reshape variables without vertical dimension to (nprofiles, 1)

    ####### set up rttov class instance
    nlevels = model.nlevels
    print('compute RTTOV for nprofiles=', nprofiles, ' with nlevels=', nlevels)
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)

    # surface data = lowest model level data
    fetch = 1e5*np.ones(nprofiles)  # default
    skintemp = model.vars_2d['skintemp']
    sfc_vapor = model.vars_3d['vapor'][:,-1]  # surface water vapor
    terrain = model.vars_2d['terrain'].ravel()/1000.  # convert to km
    sfc_p_t_qv_u_v_fetch = np.stack([model.vars_2d['psfc'], 
                                     skintemp, sfc_vapor, 
                                     model.vars_2d['u10'], 
                                     model.vars_2d['v10'], 
                                     fetch], axis=1).astype(np.float64)
    myProfiles.S2m = sfc_p_t_qv_u_v_fetch
    

    myProfiles.GasUnits = 1  # kg/kg for mixing ratios   [ ppmv_dry=0, kg_per_kg=1, ppmv_wet=2 ]
    myProfiles.MmrCldAer = True  # kg/kg for clouds and aerosoles

    # input on layers; (nprofiles, nlevels)
    myProfiles.P = model.vars_3d['p']  # in Pa
    myProfiles.T = model.get_sensible_temperature()
    myProfiles.Q = model.vars_3d['vapor']  # water vapor mixing ratio kg/kg
    # myProfiles.CO2 = np.ones((nprofiles,nlevels))*3.743610E+02
    myProfiles.Cfrac = model.vars_3d['cldfrac']  # cloud fraction

    # Cloud types - concentrations in kg/kg
    # Note: the Deff scheme disregards the cloud type (see RTTOV user guide)
    rttov_water = model.vars_3d['cloud_water']
    rttov_ice = model.vars_3d['cloud_ice'] + model.vars_3d['cloud_snow']
    # modify ice for VIS! (Kostka et al., 2014)
    myProfiles.Stco = 0*rttov_water  # Stratus Continental STCO
    myProfiles.Stma = 0*rttov_water  # Stratus Maritime STMA
    myProfiles.Cucc = rttov_water  # Cumulus Continental Clean CUCC
    myProfiles.Cucp = 0*rttov_water  # Cumulus Continental Polluted CUCP
    myProfiles.Cuma = 0*rttov_water  # Cumulus Maritime CUMA
    myProfiles.Cirr = rttov_ice  # all ice clouds CIRR

    #######################################
    # Select parameterization schemes
    
    # ClwScheme
    dummy = np.ones((nprofiles, 2))
    dummy[:,0] = 2  # clw_scheme : (1) OPAC or (2) Deff scheme
    dummy[:,1] = 1  # clwde_param : currently only "1" possible
    myProfiles.ClwScheme = dummy
    myProfiles.Clwde = model.vars_3d['re_cloud']  # microns effective diameter
    
    # icecloud[2][nprofiles]: ice_scheme, idg
    icecloud = np.array([[1, 1]], dtype=np.int32)
    myProfiles.IceCloud = expand(nprofiles, icecloud)
    
    # assume effective radius as weighted average of ice and snow
    rttov_ice_iszero = (rttov_ice == 0.)
    with np.errstate(divide='ignore', invalid='ignore'):
        massfraction_snow = model.vars_3d['cloud_snow'] / rttov_ice
    re_totalice =  (1 - massfraction_snow) * model.vars_3d['re_ice'] + \
                    massfraction_snow * model.vars_3d['re_snow']
    re_totalice = np.where(rttov_ice_iszero, 60.0, re_totalice)  # default effective radius when ice mass is zero
    
    myProfiles.Icede = re_totalice  # microns effective diameter

    # RTTOV requires time per observation as 6 integers:
    # year, month, day, hour, minute, second
    # => shape (nprofiles, 6)
    datetimes = np.zeros((nprofiles, 6), dtype=np.int32)
    print(obs_time.shape)
    for i in range(len(obs_time)):
        t = obs_time[i].astype(dt.datetime)
        datetimes[i,:] = np.array([t.year, t.month, t.day, t.hour, t.minute, t.second], dtype=np.int32)
    myProfiles.DateTimes = datetimes

    # compute solar angles from latitude, longitude (sunzen, sunazi)
    sunzen, sunazi = get_solarangles(obs_time, lat, lon, altitude=terrain*1000)
    # stack angles to (nprofiles, 4)
    myProfiles.Angles = np.stack([satzen, satazi, sunzen, sunazi], axis=1).astype(np.float64)

    # stack to (nprofiles, 3): lat, lon, elev
    myProfiles.SurfGeom = np.stack([lat, lon, terrain], axis=1).astype(np.float64)
    
    # surftype[2][nprofiles]: surftype, watertype
    surftype = np.array([[0, 0]], dtype=np.int32)
    myProfiles.SurfType = expand(nprofiles, surftype)
    
    # skin[10][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5, specularity
    # skin T has no effect on IR108, WV73 as far as my tests went
    skin = np.array([[270., 35., 0., 0., 3.0, 5.0, 15.0, 0.1, 0.3]], dtype=np.float64)
    myProfiles.Skin = expand(nprofiles, skin)
    myProfiles.Skin[:,0] = skintemp
    instrument.Profiles = myProfiles

    #######################################
    # Load IR emissivity/BRDF atlas
    try:
        assumed_month = obs_time[0].astype(dt.datetime).month
        irAtlas.loadIrEmisAtlas(assumed_month, inst=instrument, ang_corr=True)
    except pyrttov.RttovError as e:
        # If there was an error the emissivities/BRDFs will not have been modified so it
        # is OK to continue and call RTTOV with calcemis/calcrefl set to TRUE everywhere
        sys.stderr.write("Error calling atlas: {!s}".format(e))

    # Call the RTTOV direct model for each instrument:
    # no arguments are supplied to runDirect so all loaded channels are simulated
    try:
        t = time.time()
        instrument.runDirect()
        t = time.time()-t
        print('runDirect() took', int(t*10)/10., 's')
    except pyrttov.RttovError as e:
        raise RuntimeError("Error running RTTOV: {!s}".format(e))

    # print('output shape:', (instrument.BtRefl.shape))
    if instrument.RadQuality is not None:
        print('Quality (qualityflag>0, #issues):', np.sum(instrument.RadQuality > 0))

    dsout = xr.Dataset(
        coords = {
            'time': (("obs",), obs_time), # remove timezone for xarray compatibility
            'lat': (("obs",), lat.astype(np.float64)),
            'lon': (("obs",), lon.astype(np.float64))
        })

    for i, name in enumerate(config["instrument"]['channel_names_in_output']):
        data = instrument.BtRefl[:,i].astype(np.float32)
        if model.invalid_mask is not None:
            data[model.invalid_mask] = np.nan  # model points too far from obs -> set to nan
        dsout[name] = (("obs",), data)
    return dsout.dropna("obs")


############## CONFIGURATION

def setup_instrument_atlas(config):
    """Set up the instrument and atlas
    See config file
    """
    instrument = pyrttov.Rttov()
    instrument.FileCoef = '{}'.format(config["instrument"]['coef_file'])
    instrument.FileSccld = '{}'.format(config["instrument"]['sccld_file'])

    # read options from config
    for key, value in config["instrument"]["options"].items():
        setattr(instrument.Options, key, value)
        print('Set instrument option:', key, 'to', value)

    try:
        numbers = [int(id) for name, id in config['instrument']['channel_numbers'].items()]
        instrument.loadInst(numbers)
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    irAtlas = pyrttov.Atlas()
    irAtlas.AtlasPath = '{}/{}'.format(config['RTTOV_path'], "/emis_data")
    return instrument, irAtlas


if __name__ == '__main__':
    """Apply RTTOV observation operators to MPAS model output

    Two modes of operation:
    1) If no observation grid is supplied, we run RTTOV on all MPAS model cells.
    2) If supplied an observation grid, we run RTTOV for the observation locations only.
    
    The observation locations file must be a netcdf file with variables:
        lat: latitude in degrees
        lon: longitude in degrees
        sat_zen (optional): satellite zenith angle in degrees
        sat_azim (optional): satellite azimuth angle in degrees
        obs_time (optional): observation time as np.datetime64 array
            
    If optional variables are not supplied, they will be taken from 
    the configuration file.

    Usage: 
        python rttov_mpas.py <mpas_file> [--force]

    Example:
        python rttov_mpas.py /path/to/mpas_file

    Note:
        output is one file per mpas file, e.g. RTout_2008-07-30_18:00:00
    """
    # read yaml config file
    import yaml
    with open('config_mpas.yaml', 'r') as f:
        config = yaml.safe_load(f)

    sys.path.append(config['RTTOV_path']+'/wrapper')
    try:
        import pyrttov
    except ImportError as e:
        warnings.warn('Was RTTOV compiled on this exact machine?')
        raise e

    import argparse
    parser = argparse.ArgumentParser(description='Apply RTTOV to MPAS data')
    parser.add_argument('MPAS_file', type=str, help='path to MPAS file')
    parser.add_argument('output', type=str, help='path to output file')
    parser.add_argument('--obs_locations_file', type=str,
                        help='path to netcdf file with observation locations, satellite angles, and times')
    parser.add_argument('--force', action='store_true', help='overwrite existing output')
    parser.add_argument('--obs_time', type=str,
                        help='default observation time if not in obs_locations_file, format YYYY-MM-DDTHH:MM:SS')
    args = parser.parse_args()

    f_in_obs = args.obs_locations_file
    f_in_model = args.MPAS_file
    f_out = args.output
    force = args.force
    default_time = np.datetime64(dt.datetime.strptime(args.obs_time, '%Y-%m-%dT%H:%M:%S')).astype('datetime64[s]') if args.obs_time else None
    
    # do not run if output already exists
    if not force and os.path.isfile(f_out):
        print(f_out, 'already exists, not running RTTOV.')
        sys.exit()
    else:
        print('running RTTOV for', f_in_model)

    t0 = time.time()
    ds_model = xr.open_dataset(f_in_model, engine='netcdf4')
    ds_model = ds_model.load()

    obs_md = ObsMetadata(f_in_obs, config, default_time=default_time)
    instrument, irAtlas = setup_instrument_atlas(config)

    dsout = call_pyrttov(ds_model, obs_md, instrument, irAtlas, config, default_time)
    ds_model.close()
    
    # add attributes
    dsout.attrs['coef_file'] = config['instrument']['coef_file']
    dsout.attrs['sccld_file'] = config['instrument']['sccld_file']
    dsout.attrs['channel_numbers'] = str(config['instrument']['channel_numbers'])
    dsout.attrs['creation_date'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    dsout.attrs['creator'] = os.environ.get('USER', 'unknown')
    
    outdir = os.path.dirname(f_out)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    dsout.to_netcdf(f_out)
    elapsed = int(time.time() - t0)
    print(f_out, 'saved, took', elapsed, 'seconds.')
