from setuptools import find_packages, setup

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='bottom-up-attention',
      version='0.9.0',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      url='https://github.com/best-of-acrv/bottom-up-attention',
      description=
      'Bottom-up attention for image captioning & visual question answering',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      package_data={'bottom_up_attention': ['*.pkl']},
      install_requires=['acrv_datasets', 'nltk'],
      entry_points={
          'console_scripts': [
              'bottom-up-attention=bottom_up_attention.__main__:main'
          ]
      },
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
