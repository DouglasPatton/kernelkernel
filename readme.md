## Machine Learning Approach to Fish Distribution Modeling in the Contiguous United States Using Electrofishing Data 
##### by Doug Patton

### Data Sources
#### distribution modeling
re-use of species occurence data and Nature Serve HUC8 distributions from Cyterski et. al. 2020. Predictor variables from StreamCat (Hill et. al. 2014?)
#### mapping datasets
NHD plus V21
[USGS National Hydrography Dataset Watershed Boundary Data](https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/NHD/National/HighResolution/GDB/)

### Running the models
#### create the environment
`conda update conda` 

`conda create -n kernel -c conda-forge python=3.8 scikit-learn=0.23.2 descartes geopandas jupyterlab matplotlib`

`conda activate kernel`

`pip install sqlitedict`

`pip install pyarrow`

