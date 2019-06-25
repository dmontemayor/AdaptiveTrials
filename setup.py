from setuptools import setup, find_packages

setup(
    name='adaptivetrials',
    version='0.1',
    packages=find_packages(exclude=['tests*', 'data*']),
    license='MIT',
    description='Adaptive trials development platform for the Center for Renal Precision Medicine at the University of Texas Health San Antonio.',
    long_description=open('README.md').read(),
    #install_requires=[''],
    url='https://github.com/dmontemayor/AdaptiveTrials',
    author='Daniel Montemayor',
    author_email='montemayord2@uthscsa.edu',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
)
