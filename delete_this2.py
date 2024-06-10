import numpy as np

pathlist = []
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N125-AR25-Scale1_20240531-222435_lastFrame30.00.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N250-AR50-Scale1_20240531-222435_lastFrame30.00.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N375-AR75-Scale1_20240531-222436_lastFrame30.00.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N500-AR100-Scale1_20240531-222436_lastFrame18.58.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N525-AR105-Scale1_20240607-203512_lastFrame9.99.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N550-AR110-Scale1_20240607-203512_lastFrame9.99.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N575-AR115-Scale1_20240607-203512_lastFrame9.99.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N600-AR120-Scale1_20240607-203512_lastFrame9.99.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N625-AR125-Scale1_20240531-222434_lastFrame11.53.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N1000-AR200-Scale1_20240603-131308_lastFrame10.17.csv')
pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/PerturbEECarrotCake5_Scaled/NonIntersectingBox-N1500-AR300-Scale1_20240603-013525_lastFrame10.09.csv')


for pth in pathlist:
    dta = np.loadtxt(pth, dtype=np.float64)  # Ensure double precision when loading
    np.savetxt(pth, dta, fmt='%.15e') 