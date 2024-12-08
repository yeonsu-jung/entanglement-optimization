# %%
import numpy as np

pth = '/Users/yeonsu/Data/export/SugarDonut/NonIntersectingBox-N005003-AR100-Scale1.txt'
dta = np.loadtxt(pth)

# %%
dta.shape
# %%
np.min(dta[:,5])