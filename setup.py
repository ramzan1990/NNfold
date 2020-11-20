from setuptools import setup, find_packages

package = "nnfold"
version = "0.1.0"

setup(name = package,
      version = version,
      description="NNfold: RNA secondary structure predictor",
      url='https://github.com/ramzan1990/NNfold',
      author = 'Ramzan Umarov',
      author_email = 'ramzan.umarov@kaust.edu.sa',
      license = 'GNU GENERAL PUBLIC LICENSE',
      packages = find_packages(),
      install_requires = [
          'apetype',
          'numpy',
          'tflearn',
          'tensorflow<2',
          'sklearn'
      ],
      extras_require = {
          'documentation':  ["sphinx"],
      },
      package_data = {
          'nnfold': [
              'data/small_classes.csv',
              'data/t_classes.csv',
              'data/test.fa'
          ]
      },
      include_package_data = True,
      zip_safe = False,
      entry_points = {
          'console_scripts': [
              'nnfold-train-local=nnfold.train_local:main',
              'nnfold-train-global=nnfold.train_global:main',
              'nnfold-predict=nnfold.predict:main'
          ],
      },
      test_suite = 'nose.collector',
      tests_require = ['nose']
)

#To install with symlink, so that changes are immediately available:
#pip install -e .
