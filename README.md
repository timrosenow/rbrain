# rbrain
MRI analysis tools for rodent brains. Pre-alpha/development version: NOT FOR GENERAL USE.
Contact Tim Rosenow (Tim.Rosenow@uwa.edu.au).

Currently only a basic UNET brain segmenter is implemented.

## Requirements
* Linux (or a Linux container) - tested on Fedora 35+ and the Tensorflow docker container.
* Python 3 + pip (tested on 3.8 and 3.10)
* For rbrain segmenter: Tensorflow installed and setup (the Tensorflow docker image works).

## Installation
This is currently installed as a wheel package, although in future I will change this to a more self-contained option.

```bash
./build.sh
pip install rbrain-core/dist/*.whl
pip install rbrain-segmenter/dist/*.whl
```

