# RTTOV-WRF-MPAS
Simulate satellite observation using RTTOV from WRF or MPAS model output

- Input: NetCDF model output
- Output: brightness temperatures or reflectances

## Usage
Always call scripts in the same python environment as was used for compiling RTTOV, otherwise import `pyrttov` will likely fail.


### MPAS
The script `rttov_mpas.py` has two modes of operation:
1) If no observation grid is supplied, we run RTTOV on all MPAS model cells.
2) If supplied an obs_locations_file, we run RTTOV for the observation locations only. The model is interpolated to the observation.
   The file should have observation locations, satellite angles, and times.
   See configuration in `config_mpas.yaml`.

Usage: `python rttov_mpas.py <MPAS_input_file> <output_file> --obs_locations_file [path] --obs_time [YYYY-MM-DDTHH:MM:SS] --force`


### WRF
The WRF script is configured for the SEVIRI instrument and less flexible.
Use configuration `config_wrf.py` and `python rttov_wrf.py /path/to/wrfout (VIS|IR|both)`
For example, `python rttov_wrf.py /somepath/wrfout_d01_2008-07-30_12:00:00 both`, 
which creates the output file `/somepath/RT_wrfout_d01_2008-07-30_12:00:00.nc` 

Option: Use `VIS` to get VIS 0.6 µm reflectance, `IR` to get WV 6.2 µm, 7.3 µm and IR 10.8 µm brightness temperature or `both` to get all channels. 
You can add or remove channels in `rttov_wrf.py`.


#### Install
1) Download and compile RTTOV from [nwpsaf.eu](https://www.nwpsaf.eu/site/software/rttov/).
2) Download RTTOV-WRF for example by running `git clone https://github.com/lkugler/RTTOV-WRF-MPAS.git`
3) Run `cd RTTOV-WRF; pip install -e .` in the command line. Dependencies will be installed by the pip command.
4) Set paths in `paths.py`
5) optional: configure the python script `rttov_wrf.py`, it sets various assumptions for radiative transfer.
6) when running `rttov_wrf.py`, ensure that you have loaded the same libraries which you used to install RTTOV. For me, this is `intel-mpi intel netcdf netcdf-fortran zlib hdf5` (on VSC: `module purge; module load intel-mpi/2019.3 intel/19.1.0 netcdf/4.7.0-intel-19.0.5.281-75t52g6 netcdf-fortran/4.4.5-intel-19.0.5.281-qye4cqn zlib/1.2.11-intel-19.1.0.166-hs6m2qh  hdf5/1.10.5-intel-19.0.5.281-qyzojtm`)

In order to run, it needs:
1) compiled RTTOV / pyrttov (the python wrapper); with the path of pyrttov in PYTHONPATH (is ensured in `rttov_wrf.py`)
2) installed RTTOV-WRF (including dependencies)
3) loaded libraries for RTTOV

## RTTOV support
For the configuration of RTTOV or in case there are errors with RTTOV, consult the documentation of RTTOV or the RTTOV-python-wrapper.
(https://nwp-saf.eumetsat.int/site/software/rttov/documentation/)
