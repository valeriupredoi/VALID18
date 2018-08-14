"""
Validation Diagnostic

This diagnostic uses two datasets (control and experiment),
applies operations on their data, and plots one against the other.
It can optionally use a number of OBS, OBS4MIPS datasets.

Adapted from esmvaltool/diag_scripts/validation.py (recently added by VP)
Adaptation for Robin and Colin :)
"""

import os
import logging

import yaml
import numpy as np

from _area_pp import area_slice
from _time_area import extract_season
from _find_pp import pp_to_cube

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import iris  # noqa
import iris.analysis.maths as imath  # noqa
import iris.quickplot as qplt  # noqa

logger = logging.getLogger(os.path.basename(__file__))


def plot_contour(cube, plt_title, file_name):
    """Plot a contour with iris.quickplot (qplot)"""
    qplt.contourf(cube, cmap='RdYlBu_r', bbox_inches='tight')
    plt.title(plt_title)
    plt.gca().coastlines()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_latlon_cubes(cube_1,
                      cube_2,
                      cfg,
                      data_names,
                      short_name,
                      obs_name=None):
    """
    Plot lat-lon vars for control, experiment, and obs

    Also plot Difference plots (control-exper, control-obs)
    cube_1: first cube (dataset: dat1)
    cube_2: second cube (dataset: dat2)
    cfg: configuration dictionary
    data_names: var + '_' + dat1 + '_vs_' + dat2
    """
    plot_name = cfg['variables'][short_name]['analysis_type'] + '_' + data_names + '.png'
    plot_title = cfg['variables'][short_name]['analysis_type'] + ': ' + data_names
    cubes = [cube_1, cube_2]

    # plot difference: cube_1 - cube_2; use numpy.ma.abs()
    diffed_cube = imath.subtract(cube_1, cube_2)
    plot_contour(
        diffed_cube, 'Difference ' + plot_title,
        os.path.join(cfg['variables'][short_name]['plot_dir'],
                     'Difference_' + plot_name))

    # plot each cube
    var = data_names.split('_')[0]
    if not obs_name:
        cube_names = [data_names.split('_')[1], data_names.split('_')[3]]
        for cube, cube_name in zip(cubes, cube_names):
            plot_contour(
                cube, cube_name + ' ' +
                cfg['variables'][short_name]['analysis_type'] + ' ' + var,
                os.path.join(cfg['variables'][short_name]['plot_dir'],
                             cube_name + '_' + var + '.png'))
    else:
        # obs is always cube_2
        plot_contour(
            cube_2, obs_name + ' ' +
            cfg['variables'][short_name]['analysis_type'] + ' ' + var,
            os.path.join(cfg['variables'][short_name]['plot_dir'],
                         obs_name + '_' + var + '.png'))


def plot_zonal_cubes(cube_1, cube_2, cfg, plot_data, short_name):
    """Plot cubes data vs latitude or longitude when zonal meaning"""
    # xcoordinate: latotude or longitude (str)
    data_names, xcoordinate, period = plot_data
    var = data_names.split('_')[0]
    cube_names = [data_names.split('_')[1], data_names.split('_')[3]]
    lat_points = cube_1.coord(xcoordinate).points
    plt.plot(lat_points, cube_1.data, label=cube_names[0])
    plt.plot(lat_points, cube_2.data, label=cube_names[1])
    plt.title(period + ' Zonal Mean for ' + var + ' ' + data_names)
    plt.xlabel(xcoordinate + ' (deg)')
    plt.ylabel(var)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    png_name = 'Zonal_Means_' + xcoordinate + '_' + data_names + '.png'
    plt.savefig(
        os.path.join(cfg['variables'][short_name]['plot_dir'], period,
                     png_name))
    plt.close()


def apply_supermeans(ctrl, exper, obs_list):
    """Apply supermeans on data components"""
    ctrl_cube = ctrl.collapsed('time', iris.analysis.MEAN)
    logger.debug("Time-averaged control %s", ctrl_cube)
    exper_cube = exper.collapsed('time', iris.analysis.MEAN)
    logger.debug("Time-averaged experiment %s", exper_cube)
    if obs_list:
        obs_cube_list = []
        for obs in obs_list:
            obs_file = obs
            logger.info("Loading %s", obs_file)
            obs_cube = iris.load_cube(obs_file)
            obs_cube = obs_cube.collapsed('time', iris.analysis.MEAN)
            logger.debug("Time-averaged obs %s", obs_cube)
            obs_cube_list.append(obs_cube)
    else:
        obs_cube_list = None

    return ctrl_cube, exper_cube, obs_cube_list


def apply_seasons(cube):
    """Extract seaons and apply a time mean per season"""
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    season_cubes = [extract_season(cube, season) for season in seasons]
    season_meaned_cubes = [
        season_cube.collapsed('time', iris.analysis.MEAN)
        for season_cube in season_cubes
    ]

    return season_meaned_cubes


def coordinate_collapse(data_set, short_name, cfg):
    """Perform coordinate-specific collapse and (if) area slicing and mask"""
    # see what analysis needs performing
    analysis_type = cfg['variables'][short_name]['analysis_type']

    # if subset on LAT-LON
    if 'lat_lon_slice' in cfg:
        start_longitude = cfg['lat_lon_slice']['start_longitude']
        end_longitude = cfg['lat_lon_slice']['end_longitude']
        start_latitude = cfg['lat_lon_slice']['start_latitude']
        end_latitude = cfg['lat_lon_slice']['end_latitude']
        data_set = area_slice(data_set, start_longitude, end_longitude,
                              start_latitude, end_latitude)

    # if apply mask
    if '2d_mask' in cfg:
        mask_file = os.path.join(cfg['2d_mask'])
        mask_cube = iris.load_cube(mask_file)
        if 'mask_threshold' in cfg:
            thr = cfg['mask_threshold']
            data_set.data = np.ma.masked_array(
                data_set.data, mask=(mask_cube.data > thr))
        else:
            logger.warning('Could not find masking threshold')
            logger.warning('Please specify it if needed')
            logger.warning('Masking on 0-values = True (masked value)')
            data_set.data = np.ma.masked_array(
                data_set.data, mask=(mask_cube.data == 0))

    # if zonal mean on LON
    if analysis_type == 'zonal_mean':
        data_set = data_set.collapsed('longitude', iris.analysis.MEAN)

    # if zonal mean on LAT
    if analysis_type == 'meridional_mean':
        data_set = data_set.collapsed('latitude', iris.analysis.MEAN)

    # if vertical mean
    elif analysis_type == 'vertical_mean':
        data_set = data_set.collapsed('pressure', iris.analysis.MEAN)

    return data_set


def do_preamble(short_name, cfg):
    """Execute some preamble functionality"""
    # prepare output dirs
    if not os.path.exists(cfg['variables'][short_name]['plot_dir']):
        os.makedirs(cfg['variables'][short_name]['plot_dir'])
    time_chunks = ['alltime', 'DJF', 'MAM', 'JJA', 'SON']
    time_plot_dirs = [
        os.path.join(cfg['variables'][short_name]['plot_dir'], t_dir)
        for t_dir in time_chunks
    ]
    for time_plot_dir in time_plot_dirs:
        if not os.path.exists(time_plot_dir):
            os.makedirs(time_plot_dir)


def get_all_datasets(short_name, cfg):
    """Get control, exper and obs datasets"""
    ctl_suite_root = cfg['variables'][short_name]['control_model_file']
    exp_suite_root = cfg['variables'][short_name]['exper_model_file']
    obs_selection = None
    years_range = cfg['variables'][short_name]['years_range']
    years = [
        str(yr) for yr in range(int(years_range[0]), int(years_range[-1] + 1))
    ]
    c_suite_name = cfg['variables'][short_name]['control_suite_name']
    e_suite_name = cfg['variables'][short_name]['exper_suite_name']
    stream = cfg['variables'][short_name]['stream']
    stash_code = cfg['variables'][short_name]['stash_code']
    ctrl_cube = pp_to_cube(cfg['variables'][short_name]['control_suite_name'],
                           cfg['variables'][short_name]['stream'],
                           ctl_suite_root,
                           cfg['variables'][short_name]['stash_code'],
                           cfg['variables'][short_name]['path_type'], years)
    exp_cube = pp_to_cube(cfg['variables'][short_name]['exper_suite_name'],
                          cfg['variables'][short_name]['stream'],
                          exp_suite_root,
                          cfg['variables'][short_name]['stash_code'],
                          cfg['variables'][short_name]['path_type'], years)
    ctrl_cube_fname = '_'.join([
        'CTRL', c_suite_name, stream, stash_code, years[0], years[-1]
    ]) + '.nc'
    exp_cube_fname = '_'.join(
        ['EXP', e_suite_name, stream, stash_code, years[0], years[-1]]) + '.nc'
    ctrl_cube_path = os.path.join(cfg['variables'][short_name]['plot_dir'],
                                  ctrl_cube_fname)
    exp_cube_path = os.path.join(cfg['variables'][short_name]['plot_dir'],
                                 exp_cube_fname)
    iris.save(ctrl_cube, ctrl_cube_path)
    iris.save(exp_cube, exp_cube_path)

    return ctrl_cube, exp_cube, obs_selection


def plot_ctrl_exper(ctrl, exper, cfg, plot_key, short_name):
    """Call plotting functions and make plots depending on case"""
    if cfg['variables'][short_name]['analysis_type'] == 'lat_lon':
        plot_latlon_cubes(ctrl, exper, cfg, plot_key, short_name)
    elif cfg['variables'][short_name][
            'analysis_type'] == 'zonal_mean':
        plot_info = [plot_key, 'latitude', 'alltime']
        plot_zonal_cubes(ctrl, exper, cfg, plot_info, short_name)
    elif cfg['variables'][short_name][
            'analysis_type'] == 'meridional_mean':
        plot_info = [plot_key, 'longitude', 'alltime']
        plot_zonal_cubes(ctrl, exper, cfg, plot_info, short_name)


def plot_ctrl_exper_seasons(ctrl_seasons, exper_seasons, cfg, plot_key,
                            short_name):
    """Call plotting functions and make plots with seasons"""
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    if cfg['variables'][short_name]['analysis_type'] == 'zonal_mean':
        for c_i, e_i, s_n in zip(ctrl_seasons, exper_seasons, seasons):
            plot_info = [plot_key, 'latitude', s_n]
            plot_zonal_cubes(c_i, e_i, cfg, plot_info, short_name)
    elif cfg['variables'][short_name][
            'analysis_type'] == 'meridional_mean':
        for c_i, e_i, s_n in zip(ctrl_seasons, exper_seasons, seasons):
            plot_info = [plot_key, 'longitude', s_n]
            plot_zonal_cubes(c_i, e_i, cfg, plot_info, short_name)


def main(cfg):
    """Execute validation analysis and plotting"""
    logger.setLevel(cfg['log_level'].upper())

    # select variables and their corresponding obs files
    for short_name in cfg['variables'].keys():
        do_preamble(short_name, cfg)
        logger.info("Processing variable %s", short_name)

        # control, experiment and obs's and the names
        if cfg['variables'][short_name]['reuse_cubes']:
            # just get the already saved cube
            c_suite_name = cfg['variables'][short_name]['control_suite_name']
            e_suite_name = cfg['variables'][short_name]['exper_suite_name']
            stream = cfg['variables'][short_name]['stream']
            stash_code = cfg['variables'][short_name]['stash_code']
            years_range = cfg['variables'][short_name]['years_range']
            years = [
                str(yr)
                for yr in range(int(years_range[0]), int(years_range[-1] + 1))
            ]
            ctrl_cube_fname = '_'.join([
                'CTRL', c_suite_name, stream, stash_code, years[0], years[-1]
            ]) + '.nc'
            exp_cube_fname = '_'.join([
                'EXP', e_suite_name, stream, stash_code, years[0], years[-1]
            ]) + '.nc'
            ctrl_cube_path = os.path.join(
                cfg['variables'][short_name]['plot_dir'], ctrl_cube_fname)
            exp_cube_path = os.path.join(
                cfg['variables'][short_name]['plot_dir'], exp_cube_fname)
            ctrl = iris.load_cube(ctrl_cube_path)  # can add time constr
            exper = iris.load_cube(exp_cube_path)  # can add time constr
            obs = None  # empty for now
        else:
            ctrl, exper, obs = get_all_datasets(short_name, cfg)
        ctrl_name = cfg['variables'][short_name]['control_model']
        exper_name = cfg['variables'][short_name]['exper_model']
        # set a plot key holding info on var and data set names
        plot_key = short_name + '_' + ctrl_name + '_vs_' + exper_name

        # get seasons if needed then apply analysis
        if cfg['variables'][short_name]['seasonal_analysis']:
            ctrl_seasons = apply_seasons(ctrl)
            exper_seasons = apply_seasons(exper)
            ctrl_seasons = [
                coordinate_collapse(cts, short_name, cfg)
                for cts in ctrl_seasons
            ]
            exper_seasons = [
                coordinate_collapse(exps, short_name, cfg)
                for exps in exper_seasons
            ]
            plot_ctrl_exper_seasons(ctrl_seasons, exper_seasons, cfg, plot_key,
                                    short_name)

        # apply the supermeans, collapse a coord and plot
        ctrl, exper, obs_list = apply_supermeans(ctrl, exper, obs)
        ctrl = coordinate_collapse(ctrl, short_name, cfg)
        exper = coordinate_collapse(exper, short_name, cfg)
        plot_ctrl_exper(ctrl, exper, cfg, plot_key, short_name)

        # apply desired analysis on obs's
        if obs_list:
            for obs_i, obsfile in zip(obs_list, obs):
                obs_analyzed = coordinate_collapse(obs_i, short_name, cfg)
                obs_name = os.path.basename(obsfile).split('_')[1]
                plot_key = short_name + '_' + ctrl_name + '_vs_' + obs_name
                if cfg['variables'][short_name]['analysis_type'] == 'lat_lon':
                    plot_latlon_cubes(
                        ctrl,
                        obs_analyzed,
                        cfg,
                        plot_key,
                        short_name,
                        obs_name=obs_name)


# execution as shell
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
main(config)
