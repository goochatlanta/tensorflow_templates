#%%

"""
Created on Thu May  7 12:43:20 2020
@author: marko
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('/home/brian.harrington/cs4321/models/mnist_tests2022-07-22_11-23-07/log.csv', delimiter=';')
plt.plot(df['epoch'], df['loss'], 'b', linewidth= 2, label = "Training")
plt.plot(df['epoch'], df['val_loss'], 'r', linewidth= 2, label = "Validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
# %%

plt.plot(df['epoch'], df['accuracy'], 'b', linewidth= 2, label = "Training")
plt.plot(df['epoch'], df['val_accuracy'], 'r', linewidth= 2, label = "Validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()
# %%
