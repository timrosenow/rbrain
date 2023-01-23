from setuptools import setup

setup(
    name="rbrain",
    version="0.1.0",
    description="MRI analyis tools for rodent brains",
    url="https://github.com/timrosenow/rbrain",
    author='Tim Rosenow',
    author_email='Tim.Rosenow@uwa.edu.au',
    license="MPL 2.0",
    packages=['rbrain', 'rbrain.core', 'rbrain.segmenter'],
    install_requires=[
        'keras',
        'nibabel',
        'numpy',
        'Pillow',
        'scipy',
        'tensorflow',
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ]
)
