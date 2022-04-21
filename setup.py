from setuptools import setup, find_packages
import versioneer

setup(
    name="chic",
    version=versioneer.get_version(),
    description="Coarse-graining Hybrid and Inorganic Crystals",
    url='https://github.com/tcnicholas/chic',
    author='Thomas C Nicholas',
    author_email='thomas.nicholas@chem.ox.ac.uk',
    license='BSD',

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9'
                ],

    keywords='pymatgen, MOFs',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[
        'pymatgen',
        'numba',
        'networkx',
        'dscribe',
        'umap',
        'scikit-learn',
    ],
    cmdclass=versioneer.get_cmdclass(),
)
