"""Convert a list of PP files to a single iris cube/netcdf file"""
import fnmatch
import os
import numpy as np
import iris

nr_months = {
    '01': 'jan',
    '02': 'feb',
    '03': 'mar',
    '04': 'apr',
    '05': 'may',
    '06': 'jun',
    '07': 'jul',
    '08': 'aug',
    '09': 'sep',
    '10': 'oct',
    '11': 'nov',
    '12': 'dec'
}
tz_root = '01T0000Z/'


def recube(cube):
    """Rebuild the cube with time as DIM_COORD"""
    new_data = np.empty((1, cube.data.shape[0], cube.data.shape[1]))
    new_data[0] = cube.data
    new_t = cube.coord('time')
    new_lat = cube.coord('latitude')
    new_lon = cube.coord('longitude')
    coords_spec = [(new_t, 0), (new_lat, 1), (new_lon, 2)]
    new_cube = iris.cube.Cube(new_data, dim_coords_and_dims=coords_spec)

    return new_cube


def pp_to_cube(suite_name, stream, suite_rootdir, stash_code, path_style,
               years):
    """Find and convert many pp files to a single netCDF (no CMORization!)"""
    # set the file pattern
    stream_part_2 = stream[1:]
    cube_list = []
    for yr in years:
        for mo in nr_months.keys():
            fileDescriptor = suite_name + '.' + stream_part_2 \
                             + yr + nr_months[mo] + '.pp'
            if path_style == 'jasmin':
                basedir = os.path.join(suite_rootdir, yr + mo + tz_root)
            elif path_style == 'moose':
                basedir = os.path.join(
                    suite_rootdir.strip(suite_rootdir[-1]), stream + '.pp')
            elif path_style == 'mangled':
                basedir = suite_rootdir
            result = []
            for path, _, files in os.walk(basedir, followlinks=True):
                files = fnmatch.filter(files, fileDescriptor)
                if files:
                    result.extend(os.path.join(path, f) for f in files)
            if os.path.isfile(result[0]):
                stash_cons = iris.AttributeConstraint(STASH=stash_code)
                cb = iris.load(result[0], stash_cons)[0]
                cbr = recube(cb)
                cube_list.append(cbr)

    try:
        cube = iris.cube.CubeList(cube_list).concatenate_cube()
        return cube
    except iris.exceptions.ConcatenateError as ex:
        print('Can not concatenate cubes: ', ex)
        print('Differences: %s', ex.differences)
        raise ex
