from setuptools import setup

setup(
    name="confused",
    version='0.0.1-dev',
    py_modules=['confused'],
    install_requires=[
        'altair',
        'pandas'],    
    author='Bering',
    description='Confused: model performance visualisation',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',        
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ml ai',
    url='https://beringresearch.com'
    )