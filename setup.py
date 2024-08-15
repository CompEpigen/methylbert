from setuptools import setup
from setuptools import find_packages

import os, glob, warnings

warnings.filterwarnings("ignore")

setup(name='methylbert',
	version='1.0.0',

	packages=find_packages("src"),
	package_dir = {"": "src"},

	install_requires=[
		'Bio', 
		'biopython<=1.81',
		'matplotlib<3.3',
		'numpy<1.21',
		'pandas<1.4.0',
		'pysam',
		'scikit_learn<1.1.0',
		'scipy<1.7.0',
		'torch>=1.10.0',
		'tqdm',
		'transformers>=2,<2.6',
		'tokenizers<0.6.0',
		'urllib3<1.27,>=1.25.4',
		'zipp==3.13.0'
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
	
