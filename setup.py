from setuptools import setup

setup(name='methylbert',
    version='0.0.1',

	packages=['methylbert'],
	package_dir = {"": "src"},

	author='Yunhee Jeong',
	author_email='y.jeong@dkfz-heidelberg.de',

	# cli
	entry_points={
		"console_scripts": [
			"methylbert = cli:main"
		]
    }

)
