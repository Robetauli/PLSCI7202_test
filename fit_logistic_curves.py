"""
For each county/year and crop progress variable in the crop yield dataset, fit a logistic curve to the crop progress data over the course of the year.
This curve is then used to interpolate missing values for that variable. Example plots are outputted to the "plots" directory (which will be created)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Dataset files
DATASET_FILE = "combined_dataset_weekly_truncated.npz"  # Input dataset (compressed)
OUTPUT_FILE = 'combined_dataset_weekly_logistic_interpolation'  # File path to output dataset to

# Column indices of different crop types (outputs)
CROP_INDICES = list(range(2, 8))
OUTPUT_NAMES = ['corn', 'upland_cotton', 'sorghum', 'soybeans', 'spring_wheat', 'winter_wheat']

# Column indices of the progress variables for each crop
PROGRESS_INDICES_WEEKLY = {'corn': list(range(1464, 1880)),
                           'upland_cotton': list(range(2140, 2556)),
                           'sorghum': list(range(2816, 3232)),
                           'soybeans': list(range(3492, 4012)),
                           'spring_wheat': list(range(4948, 5364)),
                           'winter_wheat': list(range(5624, 6196))}
DATASET_CSV_FILE = "combined_dataset_weekly_1981-2020_truncated.csv"
COLUMN_NAMES = pd.read_csv(DATASET_CSV_FILE, index_col=False, nrows=0).columns.tolist()  # list of columns
TIME_INTERVALS = 52  # number of weeks in year

# Read in data
raw_data = np.load(DATASET_FILE)
data = raw_data['data']
print("Data shape", data.shape)
counties = data[:, 0].astype(int)
years = data[:, 1].astype(int)
Y = data[:, CROP_INDICES]
X = data[:, 8:]


# For each progress variable, fit a logistic curve. Fill in values.
# Code from https://www.kaggle.com/orianao/covid-19-logistic-curve-fitting
def logistic(x, k, x0):
    return 100 / (1 + np.exp(-k * (x - x0))) #+ 1

 # Loop through each example (year/county)
for i in range(data.shape[0]):
    # Loop through each crop
    for crop, progress_indices in PROGRESS_INDICES_WEEKLY.items():
        # Loop through each progress variable. Note that each variable takes up 52 columns (one for each week), so
        # jump by 52 every time through the loop.
        for var_start_idx in range(progress_indices[0], progress_indices[-1] + 1, TIME_INTERVALS):
            county = counties[i]
            year = years[i]
            progress_col = COLUMN_NAMES[var_start_idx]  # Name of progress column

            # X: week number
            xdata = np.arange(TIME_INTERVALS)

            # Y: progress values
            ydata = data[i, var_start_idx:var_start_idx+TIME_INTERVALS]

            # Mask of missing progress data
            missing_progress_data = np.isnan(ydata)

            # Filter X/Y to only include weeks with progress data
            xdata = xdata[~missing_progress_data]
            ydata = ydata[~missing_progress_data]
            if xdata.shape[0] < 2:  # If there are less than 2 weeks with progress data, ignore
                continue

            # Fit logistic curve to X/Y
            try:
                popt, pcov = curve_fit(logistic, xdata, ydata, p0=[1, np.mean(xdata)], method="trf", maxfev=5000)
            except RuntimeError as e:
                # If the solver couldn't find a solution (happens in some pathological cases), just skip
                print("Runtime error", e)
                print(xdata)
                print(ydata)
                continue

            # Compute estimated progress for each week using the logistic curve
            fitted_data = logistic(range(TIME_INTERVALS), *popt)

            # Fill in NA entries for this progress variable with the estimated values (given by the logistic curve)
            data[i, var_start_idx:var_start_idx+TIME_INTERVALS][missing_progress_data] = fitted_data[missing_progress_data].astype(int)

            # Plot example curves for some datapoints
            if i % 17 == 0:
                plt.title("Crop progress (blue) and logistic curve (red): year " + str(year) + ", county " + str(county))
                plt.scatter(xdata, ydata, c='b', label='data')
                plt.plot(range(TIME_INTERVALS), logistic(range(TIME_INTERVALS), *popt), 'r-', label='fit')
                plt.legend()
                county_dir = os.path.join("plots/progress_curves", str(county), str(year))
                if not os.path.exists(county_dir):
                    os.makedirs(county_dir)
                plt.savefig(os.path.join(county_dir, "progress_" + progress_col + "_county_" + str(county) + "_year_" + str(year) + ".png"))
                plt.close()

np.savez_compressed(OUTPUT_FILE, data=data)
