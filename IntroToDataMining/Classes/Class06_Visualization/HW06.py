#%%

# Settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
plt.style.use('classic')

#%%
# Loading the data
tmp = pd.read_table("GSS.dat")
tmp.to_csv("data.csv")

# Reading the csv
columnName = ["year","spevwork", "spwrkslf", "childs", "age", "educ", "sex", "income", "rincome", "sphrs2", "sphrs1", "id_", "wrkstat", "hrs1", "hrs2", "evwork", "wrkslf", "divorce", "spwrksta", "ballot"]
data = pd.read_csv("data.csv", header = None, sep = "\s+", error_bad_lines=False, names = columnName)

#%%

# Checking the head
print(data.head())

#%%

# Setting the index
data.set_index(['id_'], inplace = True)

# Testing the index is unique
data.index.is_unique

# Reset the id, because the is_unique return the false
data = data.reset_index()

#%%

# Now we start to make some plot
age = data["age"]
print(age)
plt.hist(age, label='Age',edgecolor='black', linewidth=1.2)
plt.xlabel('Ages of Respondent')
plt.ylabel('Num of Res.')
plt.show()

#%%

plt.boxplot(age)

#%%

subSex1 = data[ data['sex']==1 ]
subSex2 = data[ data['sex']==2 ]

# create a 2x3 subplot areas for contrasts
fig, axs = plt.subplots(2, 3) 

# basic plot
axs[0, 0].boxplot(subSex1['childs'])
axs[0, 0].set_title('basic plot')

# notched plot
axs[0, 1].boxplot(subSex1['childs'], 1)
axs[0, 1].set_title('notched plot')

# change outlier point symbols
axs[0, 2].boxplot(subSex1['childs'], 0, 'gD')
axs[0, 2].set_title('change outlier\npoint symbols')

# don't show outlier points
axs[1, 0].boxplot(subSex1['childs'], 0, '')
axs[1, 0].set_title("don't show\noutlier points")

# horizontal boxes
axs[1, 1].boxplot(subSex1['childs'], 0, 'rs', 0)
axs[1, 1].set_title('horizontal boxes')

# change whisker length
axs[1, 2].boxplot(subSex1['childs'], 0, 'rs', 0, 0.75)
axs[1, 2].set_title('change whisker length')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)



#%%

plt.violinplot( [ list(subSex1['age']), list(subSex2['age']), list(subSex1['childs']), list(subSex2['childs'])] , positions = [1,2,3,4] )
plt.xticks(np.arange(0,5))
plt.xlabel('Sex and Childs')
plt.ylabel('Age')


#%%
# subset
subChilds = data[data['childs'] > 1 ]
plt.plot(subChilds.age, subChilds.childs, 'o')
plt.ylabel('Childs')
plt.xlabel('Res. Age')
plt.show()

#%%
# Add jittering 
fuzzychilds = subChilds.childs + np.random.normal(0,2, size=len(subChilds.childs))
plt.plot(subChilds.age, fuzzychilds, 'o', markersize=3, alpha = 0.1)
plt.ylabel('Respondent childs')
plt.xlabel('Age')
plt.show()

#%%
# Add jittering to x as well
fuzzyage = subChilds.age + np.random.normal(0,1, size=len(subChilds.age))
plt.plot(fuzzyage, fuzzychilds, 'o', markersize=3, alpha = 0.1)
plt.ylabel('Respondent childs')
plt.xlabel('Age')
plt.show()