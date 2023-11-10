from setuptools import setup, find_packages


setup(
    name='chic-lib',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pymatgen',
        'ase',
        'networkx',
        'crystal-toolkit',
        'crystaltoolkit-extension'
    ],
    author='Thomas C Nicholas',
    author_email='thomas.nicholas@chem.ox.ac.uk',
    description='A set of tools for coarse-graining and back-mapping frameworks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tcnicholas/chic',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)

