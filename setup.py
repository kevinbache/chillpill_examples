from setuptools import find_packages
from setuptools import setup


setup(
    name="chillpill_examples",
    version='0.0.1',
    url="git@github.com:kevinbache/chillpill_examples",
    license='MIT',

    author="Kevin Bache",
    author_email="kevin.bache@gmail.com",

    description="Brown, paper, tied up with string.",
    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'keras',
        'google-cloud-storage',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
