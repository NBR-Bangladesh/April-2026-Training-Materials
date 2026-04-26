# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:52:16 2025

@author: ssj34
"""

import numpy as np
import pandas as pd
from pysfa import SFA
from pysfa.dataset import load_Tim_Coelli_frontier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline
#from stata_python import *


# filename = 'GDP Per Capita Constant 2015 USD Jul 2025.xls'
# df_gdp_pc = convert_WDI_file(filename, 'GDP_PC_Constant_USD')
# df_gdp_pc['Country_Code'] = df_gdp_pc['Country_Code'].replace(['PSE'],'WBG')
# df_gdp_pc.to_csv('GDP Per Capita Constant 2015 USD Jul 2025.csv', index=False)
# filename = 'Trade in percentage of GDP Jul 2025.xls'
# df_trade = convert_WDI_file(filename, 'Trade')
# df_trade.to_csv('Trade in percentage of GDP Jul 2025.csv', index=False)

# df_revenue_data = pd.read_csv('tax_revenue_imf_apr_2025.csv')        

#import the data from Tim Coelli Frontier 4.1
# df = load_Tim_Coelli_frontier(x_select=['labour', 'capital'],
#                               y_select=['output'])

# df = multi_merge(df_gdp_pc, [df_trade, df_revenue_data[['Country', 'Country_Code', 'year', 'Tax_Revenue']]], ['Country_Code', 'year'])

df = pd.read_csv("combined_data_stochastic_frontier.csv")

# adding 1e-2 results in convergence in the model
df['log_Tax_Revenue'] = np.log(df['Tax_Revenue'] + 1e-2)
df['log_GDP_PC'] = np.log(df['GDP_PC_Constant_USD'] + 1e-2)
df['log_Trade'] = np.log(df['Trade'] + 1e-2)

# Using pooled data for all years
# This can also be done for a specific year say ==2019
df = df[df['year']>=1990]
df = df.dropna()

y = np.log(df['Tax_Revenue']).values
x = np.log(df[['GDP_PC_Constant_USD', 'Trade']]).values

# Estimate SFA model
res = SFA.SFA(y, x, fun=SFA.FUN_PROD, method=SFA.TE_teJ)
res.optimize()
# print estimates
print(res.get_beta())
print(res.get_residuals())

# print estimated parameters
print(res.get_lambda())
print(res.get_sigma2())
print(res.get_sigmau2())
print(res.get_sigmav2())

# print statistics
print(res.get_pvalue())
print(res.get_tvalue())
print(res.get_std_err())

# OR print summary
print(res.summary())

# print TE
print(res.get_technical_efficiency())
df['tax_effort'] = res.get_technical_efficiency()
df['tax_capacity'] = df['Tax_Revenue']/df['tax_effort']

# Plot

# Filter for 2019
df_2019 = df[df['year'] == 2019].copy()

# Drop outliers LSO and MOZ
df_2019_filtered = df_2019[~df_2019['Country_Code'].isin(['LSO', 'MOZ'])].copy()

# Bin log_GDP_PC into 0.1 width bins
bin_width = 0.1
df_2019_filtered['gdp_bin'] = (df_2019_filtered['log_GDP_PC'] / bin_width).round() * bin_width

# Compute maximum tax_capacity for each bin
max_eff_by_bin = df_2019_filtered.groupby('gdp_bin')['tax_capacity'].max().reset_index()

# Smooth max efficiency frontier using spline
x_smooth = max_eff_by_bin['gdp_bin'].values
y_smooth = max_eff_by_bin['tax_capacity'].values
mask = ~np.isnan(x_smooth) & ~np.isnan(y_smooth)
x_smooth = x_smooth[mask]
y_smooth = y_smooth[mask]
sorted_idx = np.argsort(x_smooth)
x_sorted = x_smooth[sorted_idx]
y_sorted = y_smooth[sorted_idx]
spline = make_interp_spline(x_sorted, y_sorted, k=3)
x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 300)
y_fine = spline(x_fine)

# Plot everything
plt.figure(figsize=(12, 7))
plt.scatter(df_2019_filtered['log_GDP_PC'], df_2019_filtered['Tax_Revenue'], alpha=0.6, label='Observed Tax Revenue')

# Annotate countries
for _, row in df_2019_filtered.iterrows():
    plt.annotate(row['Country_Code'], (row['log_GDP_PC'], row['Tax_Revenue']),
                 textcoords="offset points", xytext=(2, 2), ha='left', fontsize=8)

# Plot smoothed max-efficiency frontier
plt.plot(x_fine, y_fine, 'k--', linewidth=2, label='Smoothed Max Efficiency Frontier')

# # Plot model-predicted average
# plt.plot(x_vals, model_line, 'r-', linewidth=2, label='Model-Predicted Frontier (Avg Trade)')

# Final formatting
plt.xlabel('Log GDP Per Capita')
plt.ylabel('Tax Revenue (% of GDP)')
plt.title('Observed Tax Revenue with Tax Capacity (2019)')
plt.legend()
plt.grid(True)
plt.tight_layout()

df = df[['Country', 'Country_Code', 'year', 'GDP_PC_Constant_USD', 'Trade', 
       'Tax_Revenue', 'tax_effort', 'tax_capacity']]
df.to_csv("tax_capacity_using_frontier.csv")

#save figure
filename='Tax Capacity - Frontier.png'
plt.savefig('Tax Capacity - Frontier.png', dpi=200, bbox_inches="tight")

plt.show()