from setuptools import setup

setup(
   name='rshdmrpy',
   version='0.1',
   description='Python implementation RS-HDMR',
   keywords='sobol sensitivity analysis',
   author='Frederick Bennett',
   author_email='frederick.bennett@des.qld.gov.au',
   packages=['rshdmrpy'],  #same as name
   install_requires=[
            'numpy',
            'pandas',
            'matplotlib',
            'scipy'
    ]
)