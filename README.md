# VALID18

This is a simple tool that allows the user to directly compare results from two different
suites (denoted by EXPERIMENT and CONTROL); this is done by reading in PP files and conevrting
them to ```iris.Cube.cube``` objects then a number of processing steps are taken. It is a straightforward
alternative to the valnote validation, with a minimal configuration file.

## Dependencies
iris and pyyaml (yaml)

## Running
To run:
 - edit config.yml according to your needs
 - ```python validation.py```

## NOTE
In ```config.yml```, select one using the config: 
set ```path_type:``` to either jasmin or moose or mangled
(magled is when all pp files are in one directory eg ```/u-ar766a/```)

The pp file path is Archer-JASMIN-like:
```$HOME/u-ar766a/18500301T0000Z/ar766a.pm1850mar.pp```

the MO one is:
```moose:/crum/u-ar766/apm.pp/ar766a.pm1850mar.pp```
