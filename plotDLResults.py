#%%
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os, re, pickle

cnnOptROC = np.load('DLResults/DL_ROC_optical_only.npy')
cnnAllROC = np.load('DLResults/DL_ROC.npy')
trfOptROC = np.load('DLResults/DL_ROC_Trf_optical.npy')

#Plot roc curves
plt.figure(figsize=(5,5))
plt.plot(cnnOptROC[0], cnnOptROC[1], label='CNN Optical Only')
plt.plot(cnnAllROC[0], cnnAllROC[1], label='CNN Optical + Age + BP + RACE')
plt.plot(trfOptROC[0], trfOptROC[1], label='Transformer Optical Only')
plt.legend(loc='lower right')
plt.show()


# %%
print(list(zip(trfOptROC[0],trfOptROC[1])))

# %%
