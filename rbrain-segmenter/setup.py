from setuptools import setup, find_namespace_packages

setup(
    name="rbrain_segmenter",
    version="0.1.0",
    description="MRI segmentation add-on for rbrain",
    url="https://github.com/timrosenow/rbrain",
    author='Tim Rosenow',
    author_email='Tim.Rosenow@uwa.edu.au',
    license="MPL 2.0",
    packages=find_namespace_packages(),
    install_requires=[
        'keras',
        'nibabel',
        'numpy',
        'Pillow',
        'scipy',
        'tensorflow',
        'rbrain'
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
