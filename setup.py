from setuptools import setup
from setuptools import find_packages

import os, glob

setup(name='methylbert',
	version='0.0.1',

	packages=find_packages("src"),
	package_dir = {"": "src"},

	install_requires=[
		'Bio', 
		'matplotlib',
		'numpy<1.21',
		'pandas',
		'pysam',
		'scikit_learn',
		'scipy',
		'torch>=1.10.0',
		'tqdm',
		'transformers>=2,<3'
	],

    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/*.py")]+ \
    		   [os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/methylbert/*.py")],

	test_suite="test",

	author='Yunhee Jeong',
	author_email='y.jeong@dkfz-heidelberg.de',

	# cli
	entry_points={
		"console_scripts": [
			"methylbert = cli:main"
		]
	}

)
	