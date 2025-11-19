import os, time, warnings
import sys
import glob
import numpy as np
import datetime as dt
import xarray as xr
from pysolar.solar import get_altitude, get_azimuth

import config_wrf
path_RTTOV = config_wrf.RTTOV
sys.path.append(path_RTTOV+'/wrapper')  # to ensure that pyrttov is importable
import pyrttov

# https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov12/ir_srf/rtcoef_msg_4_seviri_srf.html
chID_for_name = dict(VIS06=1, VIS08=2, NIR16=3, WV62=5, WV73=6, IR108=9)

class Container(object):
    pass

def my_int(a):
    if len(a)==2 and a[0]=='0':
        return int(a[1])
    else:
        return int(a)

def expand(nprofiles, input_singlecolumn):
    return np.ones((nprofiles, 1))*input_singlecolumn

def wrftime_to_datetime(xtime):
    return dt.datetime.strptime(np.datetime_as_string(xtime, unit='s'), '%Y-%m-%dT%H:%M:%S')

def add_timezone_UTC(t):
    return dt.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, tzinfo=dt.timezone.utc)

def get_wrfout_time(ds):
    time_np = ds.XTIME.values
    time_dt = add_timezone_UTC(wrftime_to_datetime(time_np))
    return time_dt

def call_pyrttov(ds, config, kappa=0):
    """Run RTTOV, return xarray Dataset of reflectance or brightness temperature

    Args:
        ds (xr.Dataset): instance returned by xr.open_dataset, select one time
        config (object): see example below

    Example:
            config = setup_VIS()  # or setup_IR()
            ds_xr_out = call_pyrttov(ds_xr_in, config)
    """
    debug = False
    time_dt = get_wrfout_time(ds)
    print(time_dt)

    surface_altitude = 0.49  # in kilometres
    # coordinates used for calculating solar angles (sunzen, sunazi)
    lat = 45.
    lon = 0.
    satzen = 45.  # satellite zenith angle
    satazi = 180.  # satellite azimuth angle

    ########### read the input file
    basetemp = 300.
    p = ds.PB/100.  # (ds.P + ds.PB)/100. # ignore perturbation pressure as this could break monotonicity in p, and RTTOV
    theta = ds.T+basetemp
    qv = ds.QVAPOR  #  Water vapor mixing ratio  kg kg-1
    qi = ds.QICE + kappa * ds.QSNOW  # see section 3a in Kostka et al. (2014)  Ice mixing ratio  kg kg-1
    qc = ds.QCLOUD  # Cloud liquid water mixing ratio  kg kg-1
    cfrac = ds.CLDFRA
    tsk = ds.TSK
    u, v = ds.U10, ds.V10
    psfc = ds.PSFC/100.
    
    try:  # in case that input is a wrfinput file (doesnt have these fields)
        albedo = ds.ALBEDO
        emissivity = ds.EMISS
    except AttributeError as e:
        warnings.warn(str(e)+' -> using default values!')
        albedo = 0.17 + 0*tsk
        emissivity = 0.985 + 0*tsk

    nlevels = p.shape[0]
    def reformat_profile(arr):
        # reshape to out-dims: (nprofiles, nlevels); convention from TOA to ground
        return np.flip(arr.values.reshape((nlevels, -1)).transpose(), axis=1)

    p = reformat_profile(p)
    theta = reformat_profile(theta)
    qv = reformat_profile(qv)
    qi = reformat_profile(qi)
    qc = reformat_profile(qc)
    cfrac = reformat_profile(cfrac)
    nprofiles = p.shape[0]

    # reshape to (nprofiles, nlevels)
    u = u.values.reshape((-1, 1))
    v = v.values.reshape((-1, 1))
    tsk = tsk.values.reshape((-1, 1))
    psfc = psfc.values.reshape((-1, 1))
    albedo = albedo.values.reshape((-1, 1))
    emissivity = emissivity.values.reshape((-1, 1))

    ##  conversion to Rttov
    kappa = 2/7
    tmp = theta*(p*1e-3)**kappa  # ignoring water vapor in conversion to dry temperature

    ####### set up rttov class instances
    seviriRttov = config.seviriRttov

    nprofiles = p.shape[0]
    nlevels = p.shape[1]
    print('nlevels:', nlevels, 'nprofiles:', nprofiles)

    # calculate position of the sun (solar angles)
    sunzen = 90. - get_altitude(lat, lon, time_dt)
    sunazi = get_azimuth(lat, lon, time_dt)
    print('solar angles: zenith=', sunzen, ', azimuth=', sunazi)

    # surface data = lowest model half level data
    fetch = 1e5*np.ones(nprofiles)  # default
    sfc_p_t_qv_u_v_fetch = np.stack([psfc[:,0], tmp[:,-1], qv[:,-1], u[:,0], v[:,0], fetch],
                                    axis=1).astype(np.float64)

    # RTTOV hard limit on input data
    qv[qv<1e-11] = 1e-11

    # Declare an instance of Profiles
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)

    myProfiles.GasUnits = 1  # kg/kg for mixing ratios   [ ppmv_dry=0, kg_per_kg=1, ppmv_wet=2 ]
    myProfiles.MmrCldAer = True  # kg/kg for clouds and aerosoles

    myProfiles.P = p
    myProfiles.T = tmp
    myProfiles.Q = qv
    clw = qc  # cloud liquid water
    ciw = qi  # cloud ice water

    # input on layers; (nprofiles, nlevels)
    # myProfiles.CO2 = np.ones((nprofiles,nlevels))*3.743610E+02

    # for MW channels
    # myProfiles.CLW = clw

    if debug:
        print('cfrac', np.min(cfrac), np.max(cfrac))
    myProfiles.Cfrac = cfrac  # cloud fraction

    # WATER CLOUDS
    dummy = np.ones((nprofiles, 2))
    dummy[:,0] = 2  # clw_scheme : (1) OPAC or (2) Deff scheme
    dummy[:,1] = 1  # clwde_param : currently only "1" possible
    myProfiles.ClwScheme = dummy

    myProfiles.Clwde = 20*np.ones((nprofiles,nlevels))  # microns effective diameter
    # Cloud types - concentrations in kg/kg
    # Note: the Deff scheme disregards the cloud type (see RTTOV user guide)
    myProfiles.Stco = 0*clw  # Stratus Continental STCO
    myProfiles.Stma = 0*clw  # Stratus Maritime STMA
    myProfiles.Cucc = clw  # Cumulus Continental Clean CUCC
    myProfiles.Cucp = 0*clw  # Cumulus Continental Polluted CUCP
    myProfiles.Cuma = 0*clw  # Cumulus Maritime CUMA
    myProfiles.Cirr = ciw  # all ice clouds CIRR

    # icecloud[2][nprofiles]: ice_scheme, idg
    icecloud = np.array([[1, 1]], dtype=np.int32)
    myProfiles.IceCloud = expand(nprofiles, icecloud)
    myProfiles.Icede = 60 * np.ones((nprofiles,nlevels))  # microns effective diameter

    myProfiles.S2m = expand(nprofiles, sfc_p_t_qv_u_v_fetch)
    t_np = np.array([[time_dt.year, time_dt.month, time_dt.day, 
                      time_dt.hour, time_dt.minute, time_dt.second]], dtype=np.int32)
    myProfiles.DateTimes = expand(nprofiles, t_np)
    # angles[4][nprofiles]: satzen, satazi, sunzen, sunazi
    angles = np.array([[satzen,  satazi, sunzen, sunazi]], dtype=np.float64)
    myProfiles.Angles = expand(nprofiles, angles)

    # surfgeom[3][nprofiles]: lat, lon, elev
    surfgeom = np.array([[lat, lon, surface_altitude]], dtype=np.float64)
    myProfiles.SurfGeom = expand(nprofiles, surfgeom)
    # surftype[2][nprofiles]: surftype, watertype
    surftype = np.array([[0, 0]], dtype=np.int32)
    myProfiles.SurfType = expand(nprofiles, surftype)
    # skin[10][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5, specularity
    # skin T has no effect on IR108, WV73 as far as my tests went
    skin = np.array([[270., 35., 0., 0., 3.0, 5.0, 15.0, 0.1, 0.3]], dtype=np.float64)
    myProfiles.Skin = expand(nprofiles, skin)
    myProfiles.Skin[:,0] = tsk[:,0]

    seviriRttov.Profiles = myProfiles

    #######################################

    if True:  # Custom values
        
        # Set up the surface emissivity/reflectance arrays and associate with the Rttov objects
        surfemisrefl_seviri = np.zeros((5, nprofiles, config.nchan), dtype=np.float64)
        surfemisrefl_seviri[0,:,:] = -1  # emissivity
        surfemisrefl_seviri[1,:,:] = -1  # albedo/np.pi  # reflectance
        surfemisrefl_seviri[2,:,:] = 0.  # diffuse reflectance
        surfemisrefl_seviri[3,:,:] = 0.  # specularity
        surfemisrefl_seviri[4,:,:] = -1  # effective Tsfc
        seviriRttov.SurfEmisRefl = surfemisrefl_seviri
    else:
        # Surface emissivity/reflectance arrays must be initialised *before every call to RTTOV*
        # Negative values will cause RTTOV to supply emissivity/BRDF values (i.e. equivalent to
        # calcemis/calcrefl TRUE - see RTTOV user guide)
        # Call emissivity and BRDF atlases
        try:
            # Do not supply a channel list for SEVIRI: this returns emissivity/BRDF values for all
            # *loaded* channels which is what is required

            irAtlas = config.irAtlas
            irAtlas.loadIrEmisAtlas(time_dt.month, ang_corr=True) # Include angular correction, but do not initialise for single-instrument
            surfemisrefl_seviri[:,:,:2] = irAtlas.getEmisBrdf(seviriRttov)

            try:
                brdfAtlas = config.brdfAtlas
                brdfAtlas.loadBrdfAtlas(time_dt.month, seviriRttov) # Supply Rttov object to enable single-instrument initialisation
                brdfAtlas.IncSea = False  # Do not use BRDF atlas for sea surface types
            except AttributeError:
                pass 
            surfemisrefl_seviri[:,:,2] = brdfAtlas.getEmisBrdf(seviriRttov)
        except pyrttov.RttovError as e:
            # If there was an error the emissivities/BRDFs will not have been modified so it
            # is OK to continue and call RTTOV with calcemis/calcrefl set to TRUE everywhere
            sys.stderr.write("Error calling atlas: {!s}".format(e))

    # Call the RTTOV direct model for each instrument:
    # no arguments are supplied to runDirect so all loaded channels are simulated
    try:
        t = time.time()
        seviriRttov.runDirect()
        t = time.time()-t
        print('took', int(t*10)/10., 's')
    except pyrttov.RttovError as e:
        sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))

    # print('output shape:', (seviriRttov.BtRefl.shape))
    if seviriRttov.RadQuality is not None:
        print('Quality (qualityflag>0, #issues):', np.sum(seviriRttov.RadQuality > 0))

    dsout = ds.copy()
    delvars = [a for a in list(ds.variables) if not a in ['XLAT', 'XLONG']]  # 'OLR',
    dsout = dsout.drop_vars(delvars)

    if debug:
        mmin = data.ravel().min()
        mmax = data.ravel().max()
        print(name, 'min:', mmin, 'max:', mmax)

    nx, ny = len(ds.west_east), len(ds.south_north)
    for i, name in enumerate(config.chan_names):
        data = seviriRttov.BtRefl[:,i]
        dsout[name] = (("south_north", "west_east"),
                        data.reshape(ny, nx).astype(np.float32))

    dsout.coords['time'] = time_dt.replace(tzinfo=None)  # .strftime('%Y-%m-%d_%H:%M')

    drop_lat_lon = True  # switch to False to retain coordinates from input file.
    if drop_lat_lon:
        dsout = dsout.drop_vars(['XLAT', 'XLONG'])
    return dsout


############## CONFIGURATION

def setup_IR(ir_channels_names):
    config = Container()
    seviriRttov = pyrttov.Rttov()

    selected = []
    for name in ir_channels_names:
        selected.append(chID_for_name[name])
    config.chan_names = ir_channels_names 
    config.nchan = len(selected)
    chan_list_seviri = tuple(selected)

    seviriRttov.FileCoef = '{}/{}'.format(path_RTTOV,
                            "/rtcoef_rttov13/rttov13pred54L/rtcoef_msg_4_seviri_o3co2.dat")
    seviriRttov.FileSccld = '{}/{}'.format(path_RTTOV,
                            "/rtcoef_rttov13/cldaer_visir/sccldcoef_msg_4_seviri.dat")

    seviriRttov.Options.StoreRad = False
    seviriRttov.Options.Nthreads = 48
    seviriRttov.Options.NprofsPerCall = 1024

    seviriRttov.Options.AddInterp = True
    seviriRttov.Options.AddSolar = False   # true with MFASIS
    seviriRttov.Options.AddClouds = True
    seviriRttov.Options.GridBoxAvgCloud = True
    seviriRttov.Options.UserCldOptParam = False
    seviriRttov.Options.VisScattModel = 1  # MFASIS=3  / 1 for IR sim necessary!
    seviriRttov.Options.IrScattModel  = 2
    seviriRttov.Options.OzoneData = False
    seviriRttov.Options.VerboseWrapper = False
    seviriRttov.Options.Verbose = False  # False: do not print warnings

    # ApplyRegLimits=True: Input profiles can be clipped to the regression limits when the limits are exceeded
    seviriRttov.Options.ApplyRegLimits = True

    try:
        seviriRttov.loadInst(chan_list_seviri)
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    irAtlas = pyrttov.Atlas()
    irAtlas.AtlasPath = '{}/{}'.format(path_RTTOV, "/emis_data")

    #brdfAtlas = pyrttov.Atlas()
    #brdfAtlas.AtlasPath = '{}/{}'.format(path_RTTOV, "/brdf_data")

    config.seviriRttov = seviriRttov
    config.irAtlas = irAtlas
    #config.brdfAtlas = brdfAtlas
    return config

def setup_VIS(vis_channels_names):
    config = Container()
    seviriRttov = pyrttov.Rttov()

    selected = []
    for name in vis_channels_names:
        selected.append(chID_for_name[name])
    config.chan_names = vis_channels_names
    config.nchan = len(selected)
    chan_list_seviri = tuple(selected)

    # Set the options for each Rttov instance:
    # - the path to the coefficient file must always be specified
    # - turn RTTOV interpolation on (because input pressure levels differ from
    #   coefficient file levels)
    # - set the verbose_wrapper flag to true so the wrapper provides more
    #   information
    # - enable solar simulations for SEVIRI
    # - enable CO2 simulations for HIRS (the CO2 profiles are ignored for
    #   the SEVIRI and MHS simulations)

    seviriRttov.FileCoef = '{}/{}'.format(path_RTTOV,
                                          "/rtcoef_rttov13/rttov13pred54L/rtcoef_msg_4_seviri_o3co2.dat")
    # CLOUD COEFFICIENTS
    seviriRttov.FileSccld = '{}/{}'.format(path_RTTOV,
                                          "/rtcoef_rttov13/cldaer_visir/sccldcoef_msg_4_seviri.dat")
    # MFASIS LOOKUPTABLE
    # seviriRttov.FileMfasisCld = '{}/{}'.format(path_RTTOV,
    #                                       "/rtcoef_rttov12/mfasis_lut/rttov_mfasis_cld_msg_4_seviri_opac.H5")
    seviriRttov.FileMfasisCld = '{}/{}'.format(path_RTTOV,
                                          "/rtcoef_rttov13/mfasis_lut/rttov_mfasis_cld_msg_4_seviri_deff.H5")

    seviriRttov.Options.StoreRad = False
    seviriRttov.Options.Nthreads = 48
    seviriRttov.Options.NprofsPerCall = 1024

    seviriRttov.Options.AddInterp = True
    seviriRttov.Options.AddSolar = True
    seviriRttov.Options.AddClouds = True
    seviriRttov.Options.GridBoxAvgCloud = True
    seviriRttov.Options.UserCldOptParam = False
    seviriRttov.Options.VisScattModel = 3  # MFASIS=3
    seviriRttov.Options.IrScattModel  = 2
    seviriRttov.Options.OzoneData = False
    seviriRttov.Options.VerboseWrapper = False  #True
    seviriRttov.Options.Verbose = False  # False: do not print warnings

    # ApplyRegLimits=True: Input profiles can be clipped to the regression limits when the limits are exceeded
    seviriRttov.Options.ApplyRegLimits = True

    # Load the instruments: for HIRS and MHS do not supply a channel list and
    try:
        seviriRttov.loadInst(chan_list_seviri)
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    irAtlas = pyrttov.Atlas()
    irAtlas.AtlasPath = '{}/{}'.format(path_RTTOV, "/emis_data")

    brdfAtlas = pyrttov.Atlas()
    brdfAtlas.AtlasPath = '{}/{}'.format(path_RTTOV, "/brdf_data")

    config.seviriRttov = seviriRttov
    config.irAtlas = irAtlas
    config.brdfAtlas = brdfAtlas
    return config


if __name__ == '__main__':
    """Converts wrfout to netcdf of brightness temperature/reflectance

    Usage: 
        python rttov_wrf.py <wrfout_path> <channel_names> [--kappa=0.1] [--force]

    Example:
        python rttov_wrf.py /path/to/wrfout_d01 VIS06,WV73 --kappa=0.1

    Note:
        output is one file per wrfout file, e.g. RTout_2008-07-30_18:00:00
    """
    # which package to use for parsing command line arguments?
    import argparse
    parser = argparse.ArgumentParser(description='Converts wrfout to netcdf of brightness temperature/reflectance')
    parser.add_argument('wrfout_path', type=str, help='path to wrfout file')
    parser.add_argument('channel_names', type=str, help='comma-separated list of channel names, VIS06, WV73, IR108')
    parser.add_argument('--kappa', type=float, default=0.1, help='fraction of QSNOW to be added to QICE')
    parser.add_argument('--force', action='store_true', help='overwrite existing output')
    args = parser.parse_args()
    
    wrfout_path = args.wrfout_path
    channel_names = args.channel_names.split(',')
    kappa = float(args.kappa)
    print('wrfout_path:', wrfout_path)
    print('channel_names:', channel_names)
    
    # do not run if output already exists
    folder = os.path.dirname(wrfout_path)
    if folder == '':
        folder = '.'
    fout = folder+'/RT_'+os.path.basename(wrfout_path)+'.nc'
    if not args.force and os.path.isfile(fout):
        print(fout, 'already exists, not running RTTOV.')
        sys.exit()
    else:
        print('running RTTOV for', wrfout_path)
        
    ds = xr.open_dataset(wrfout_path, engine='netcdf4')
    times = ds.Time
    
    # calculate VIS?
    do_VIS = False
    for name in channel_names:
        if name in ['VIS06', 'VIS08', 'NIR16']:
            do_VIS = True
            break
    
    # calculate IR?
    do_IR = False
    for name in channel_names:
        if name in ['IR108', 'WV62', 'WV73']:
            do_IR = True
            break
        
    # which channels to compute?
    ir_channels_names = []
    vis_channels_names = []
    for name in channel_names:
        chan = chID_for_name[name]
        if chan >= 4:
            ir_channels_names.append(name)
        if chan < 4:
            vis_channels_names.append(name)
            
    t0 = time.time()
    ds = ds.load()
    l2_out = []
    for t in times:
        l_out = []
        
        if do_IR:
            config = setup_IR(ir_channels_names)
            out = call_pyrttov(ds.sel(Time=t), config, kappa=1)
            l_out.append(out)

        if do_VIS:
            config = setup_VIS(vis_channels_names)
            out = call_pyrttov(ds.sel(Time=t), config, kappa=kappa)
            l_out.append(out)

        dsout1 = xr.merge(l_out)
        l2_out.append(dsout1)

    ds.close()
    dsout = xr.concat(l2_out, dim='time')
    
    # add attributes for kappa
    dsout.attrs['KAPPA_VIS'] = str(kappa)
    
    dsout.to_netcdf(fout)
    elapsed = int(time.time() - t0)
    print(fout, 'saved, took', elapsed, 'seconds.')
