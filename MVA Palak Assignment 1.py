#!/usr/bin/env python
# coding: utf-8

# In[267]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import math
import matplotlib.mlab as mlab
from scipy.stats import norm
import statistics 

HAR= pd.read_csv(r'C:\Users\91964\Desktop\haryana.csv')
print (HAR)


# 1) You are required to draw a graph how the numbers have increased overtime for the
# state assigned to you. The analysis has to be conducted for the following variables (i)
# Total number of cases (ii) Total number of deceased (Hint: Remember the data
# provided to you contains daily confirmed/deceased cases NOT total
# confirmed/deceased cases)

# In[268]:


HAR = HAR[["Date_YMD", "Status", "HR"]]
HAR = HAR.loc[HAR.Status == "Confirmed"]
HAR = HAR.reset_index()
HAR['Total Confirmed']=HAR['HR'].cumsum()
HAR['Date'] = pd.to_datetime(HAR['Date_YMD'])
HAR = HAR[["Date", "Total Confirmed"]]
HAR.head()


# In[269]:


HAR.shape


# In[270]:


plt.plot_date(HAR['Date'], HAR['Total Confirmed'], linestyle='solid')
plt.xlabel('Date')
plt.ylabel('Total Confirmed Cases')
plt.title('Total Confirmed Cases of COVID-19 in Haryana')
plt.show()


# In[271]:


HAR= pd.read_csv(r'C:\Users\91964\Desktop\haryana.csv')
HAR = HAR[["Date_YMD", "Status", "HR"]]
HAR = HAR.loc[HAR.Status == "Deceased"]
HAR = HAR.reset_index()
HAR['Total Deceased']=HAR['HR'].cumsum()
HAR['Date'] = pd.to_datetime(HAR['Date_YMD'])
HAR = HAR[["Date", "Total Deceased"]]
HAR.head()


# In[272]:


plt.plot_date(HAR['Date'], HAR['Total Deceased'], linestyle='solid')
plt.xlabel('Date')
plt.ylabel('Total Deceased Cases')
plt.title('Total Deceased Cases of COVID-19 in Haryana')
plt.show()


# 2) You are required to check whether Benford’s Law apply for the state assigned to you
# on the following variables (i) Total number of cases (ii) Total number of deceased?

# In[273]:


pip install benfordslaw


# In[274]:


from benfordslaw import benfordslaw
import pandas as pd

# Initialize
bl = benfordslaw(alpha=0.05)

HAR= pd.read_csv(r'C:\Users\91964\Desktop\haryana.csv')
print (HAR)
# Extract total information.
HAR['Total Deceased']=HAR['HR'].cumsum()
A = HAR['Total Deceased'].values

# Print
print(A)

# Make fit
results = bl.fit(A)

# Plot
bl.plot(title='HARYANA')


# In[275]:


from benfordslaw import benfordslaw
import pandas as pd

# Initialize
bl = benfordslaw(alpha=0.05)

HAR= pd.read_csv(r'C:\Users\91964\Desktop\haryana.csv')
print (HAR)


# Extract total information.
HAR['Total Confirmed']=HAR['HR'].cumsum()
B = HAR['Total Confirmed'].values

# Print
print(B)

# Make fit
results = bl.fit(B)

# Plot
bl.plot(title='HARYANA')


# # ASSIGNMENT 2

# A PART

# In[276]:


HAR= pd.read_csv(r'C:\Users\91964\Desktop\haryana.csv')
HAR.head()


# In[277]:


plt.hist(HAR['confirmed'])
plt.title('Histogram of Total Confirmed')
plt.show()


# In[278]:


HAR.confirmed.plot(kind='kde', title='Kernel Density of Total Confirmed')


# In[279]:


plt.hist(HAR['deceased'])
plt.title('Histogram of Total deceased')
plt.show()


# In[280]:


HAR.deceased.plot(kind='kde', title='Kernel Density of Total Deceased')


# B PART

# In[281]:


HAR= pd.read_csv(r'C:\Users\91964\Desktop\haryana.csv')
HAR.head()


# In[282]:


import numpy as np

# Sample from a normal distribution using numpy's random number generator


# Compute a histogram of the sample
bins = np.linspace(-5, 5, 30)
histogram, bins = np.histogram(HAR['confirmed'], bins=bins, density=True)

bin_centers = 0.5*(bins[1:] + bins[:-1])
bins1 = np.linspace(-5, 5, 30)
histogram1, bins1 = np.histogram(HAR['deceased'], bins=bins, density=True)

bin_centers1 = 0.5*(bins1[1:] + bins1[:-1])

# Compute the PDF on the bin centers from scipy distribution object
from scipy import stats
pdf = stats.norm.pdf(bin_centers)
pdf1=stats.norm.pdf(bin_centers1)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, histogram, label="Histogram of samples(confirmed)")
plt.plot(bin_centers1,histogram1, label='Histogram of samples(deceased)')
plt.plot(bin_centers, pdf, label="PDF")
plt.plot(bin_centers1,pdf1, label='PDF1')
plt.legend()
plt.show()


# C PART

# In[289]:


HAR.describe()


# In[286]:


std1 = HAR['confirmed'].between(802.337313-)


# less than 68% not normally distributed as avg not defined so failed

# In[252]:


deceased_std=np.std(HAR['deceased'])
deceased_std


# In[253]:


deceased_avg=np.average(HAR['deceased'])
deceased_avg


# less than 68% not normally distributed as avg not defined so failed

# D PART

# In[264]:


import statsmodels.api as sm 
import pylab as py


# In[265]:


sm.qqplot(HAR['confirmed'], line ='45') 
py.show()


# In[266]:


sm.qqplot(HAR['deceased'], line ='45') 
py.show()


# In[ ]:




