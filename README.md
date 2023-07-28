Getting started with Gelsight sensor traditional CV processing

## Marker Tracking

Adapted from https://github.com/GelSight/tracking

## Files

- makefile: to compile library
- requirements.txt: for pip install
- Scripts:
  - `A_main.py`: where you run the main
  - `A_utility.py`: functions to process the images
  - `settings.py`: where you adjust the parameters for marker tracking
  - `WebcamCheck.py`: loop over camera devices

## Requirement

- opencv
- pybind11
- numpy

```
pip install -r requirement.txt
```

## Run make file to compile the C++ library with Python binding

```
make
```

- This gives you the file `find_marker.so` needed to track marker
