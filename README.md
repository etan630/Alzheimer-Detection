# Alzheimers Project

喵

## Setting up for windows gpu

make new environment with python 3.9

run:

`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

`pip install "tensorflow<2.11" `

`conda install --yes --file requirements.txt`

`pip install numpy --upgrade`

this is because scikit image, tensorflow, and numpy don't really like each other together.
There are a couple of weird version conflicts they have to deal with. If we don't need to use
scikit image, then we can just not have it. 