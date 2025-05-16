import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  
import scipy
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.optimize import minimize
import math
from scipy.optimize import Bounds
from scipy.stats import poisson
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import pacf
import matplotlib.dates as mdates
# import mpl_scatter_density # adds projection='scatter_density'

# os.chdir('BEE5850_Project_Fenya_Arron')

# Load data
data_dir = os.path.join("data", "EnergyUsageClassroomsAll.csv")
energy_data = pd.read_csv(data_dir)
energy_data[['Time','Zone']]= energy_data['ts'].str.split(' ', n=1, expand=True)
energy_data = energy_data.set_index('Time')
energy_data.index = pd.to_datetime(energy_data.index, format='mixed')

# Leaving out 2024 to use as testing data
# training_years = [(energy_data.index[i].year == 2022) | (energy_data.index[i].year == 2023) for i in range(energy_data.shape[0])]
# energy_data = energy_data[(training_years)]

data_dir = os.path.join("data", "2022WeatherHistorical.csv")
weather_data1 = pd.read_csv(data_dir)
weather_data1[['Time','Zone']]= weather_data1['Timestamp'].str.split(' ', n=1, expand=True)
weather_data1 = weather_data1.set_index('Time')
weather_data1.index = pd.to_datetime(weather_data1.index, format='mixed')
data_dir = os.path.join("data", "2023WeatherHistorical.csv")
weather_data2 = pd.read_csv(data_dir)
weather_data2[['Time','Zone']]= weather_data2['Timestamp'].str.split(' ', n=1, expand=True)
weather_data2 = weather_data2.set_index('Time')
weather_data2.index = pd.to_datetime(weather_data2.index, format='mixed')
data_dir = os.path.join("data", "WeatherHistorical.csv")
weather_data3 = pd.read_csv(data_dir)
weather_data3[['Time','Zone']]= weather_data3['Timestamp'].str.split(' ', n=1, expand=True)
weather_data3 = weather_data3.set_index('Time')
weather_data3.index = pd.to_datetime(weather_data3.index, format='mixed')
weather_data = pd.concat([weather_data1, weather_data2, weather_data3])
weather_data = weather_data.drop_duplicates()


col = 'Ithaca, NY, United States Humidity'
weather_data[col] = (weather_data[col] - weather_data[col].mean())/weather_data[col].std()

col = 'Ithaca, NY, United States Temp'
weather_data[col] = (weather_data[col] - weather_data[col].mean())/weather_data[col].std()

weather_data = weather_data.drop(weather_data.index.difference(energy_data.index))
energy_data = energy_data.drop(energy_data.index.difference(weather_data.index))
energy_data = energy_data.drop(['ts','Zone'], axis=1)

data_dir = os.path.join("data", "Cornell_Classroom_Building_Square_Footage.xlsx - Sheet1.csv")
square_footage = pd.read_csv(data_dir)
square_footage['Building Name'] = square_footage['Building Name'].replace(" ", "")
square_footage['Building Name'] = [square_footage['Building Name'][i].replace(" ", "") for i in range(square_footage.shape[0])]
square_footage.index = square_footage['Building Name']

energy_data = energy_data.drop(columns=['KimballHall','StatlerHall','VetMedicalCenter','PlantScience','KlarmanHall','GatesHall','MyronTaylorHall','GoldwinSmithHall','UrisHall','StimsonHall','MarthaVanRensselaerComplex','SageHall'])

# Calculate z score
for bdg in energy_data.columns:
    energy_data[bdg] = (energy_data[bdg] -energy_data[bdg].mean())/energy_data[bdg].std()
    
np.isnan(weather_data['Ithaca, NY, United States Humidity']).any()
y = energy_data.transpose().to_numpy().reshape(-1, 1)
nan_data = (np.isnan(y))
y =y[~nan_data].reshape(-1, 1)  


humid = np.tile(np.array(weather_data.iloc[:,[3]]),(energy_data.shape[1],1)).reshape(-1, 1)[~nan_data].reshape(-1, 1)  
temp = np.tile(np.array(weather_data.iloc[:,[4]]),(energy_data.shape[1],1)).reshape(-1, 1)[~nan_data].reshape(-1, 1) 
# sites = np.tile(df.columns, df.shape[0])

# x_lat =x_lat.reshape(-1, 1)[~nan_data].reshape(-1, 1)
# x_lon =x_lon.reshape(-1, 1)[~nan_data].reshape(-1, 1)
# dates = energy_data.index
dates = np.tile(energy_data.index, energy_data.shape[1])
months = np.zeros(len(dates))
hours = np.zeros(len(dates))
# len(dates)
# Store months for stratification 
for i in range(dates.shape[0]):
    ts = pd.Timestamp(dates[i])
    months[i]=ts.month

hours = np.zeros(dates.shape[0])
for i in range(dates.shape[0]):
    ts = pd.Timestamp(dates[i])
    hours[i]=ts.hour


months = months.reshape(-1, 1)[~nan_data].reshape(-1, 1)  
hours = hours.reshape(-1, 1)[~nan_data].reshape(-1, 1)  
X = np.concatenate((humid, temp), axis=1)
# pd.Series(y.reshape(-1,))[70000:70300].plot()

# Electricity use regression model
def electricity_linear_model(params, X):
    μ =  params[0] + X[:,0]* params[1] + X[:,1]* params[2] + params[3]*np.cos((params[4]*hours).reshape(-1,) + params[5]) + params[6]*np.cos((params[7]*months).reshape(-1,) + params[8])
    return μ

# return logliklihood for model, assuming IID residuals 
def electricity_demand_model(params, X, y):
    σ = params[-1]
    μ = electricity_linear_model(params, X)
    ll = np.sum(norm.logpdf(pd.Series(y.reshape(-1,)), μ, scale=σ))  # compute log-likelihood
    return ll

# Set bounds and initial values
lb = [-10.0, -10.0, -10.0, -10.0, -100.0, 0, -100.0, -100.0, 0, 0.0001]
ub = [10.0, 10.0, 10.0, 10.0, 100.0, 2*math.pi, 100.0, 1000.0, 2*math.pi, 10.0]
init = [0.1, 0.1, 0.1, 0.01, 0.1, 0.1, 0.1,0.1, 0.1, 0.1]

# Fit the model by minimizing negative of log likelihood
result = minimize(lambda θ: -electricity_demand_model(θ, X, y), init, bounds=list(zip(lb, ub)))
θ_mle = result.x
pd.DataFrame(θ_mle).to_csv('params_classrooms_2022_2024_hour_month_noAR1_ZScore_May4.csv')

# Calculate AIC for IID model
AIC_IID = -2*(electricity_demand_model(θ_mle, X, y) - len(θ_mle))


# Use whitening method, get logliklihood for model with AR(1) Residuals

def elec_loglik(params, X, y):
    ρ = params[-1]
    σ = params[-2]
    elec_sim = electricity_linear_model(params, X)
    residuals = (y - np.array(elec_sim).reshape(-1,1))
    # ll = np.sum(norm.logpdf(gmsl_data, loc=y, scale=np.sqrt(σ**2+gmsl_error**2)))  # compute log-likelihood
    T = len(y)
    ll = 0  # initialize log-likelihood counter
    for t in range(len(elec_sim)):
        if t == 0:
            ll += norm.logpdf(residuals[0], loc=0, scale=np.sqrt(σ**2 / (1 - ρ**2)))
        else:
            resid_wn = residuals[t] - ρ * residuals[t-1]
            ll += norm.logpdf(resid_wn, loc=0, scale=np.sqrt(σ**2))

    return ll
lb = [-10.0, -10.0, -10.0, -10.0, -100.0, 0, -100.0, -100.0, 0, 0.0001, -0.99]
ub = [10.0, 10.0, 10.0, 10.0, 100.0, 2*math.pi, 100.0, 1000.0, 2*math.pi, 10.0, 0.99]
init = list(θ_mle)+ [0.1]

# Minimize negative of logliklihood to fit the model, this will take a long time
result = minimize(lambda θ: -elec_loglik(θ, X, y), init, bounds=list(zip(lb, ub)))
θ_mle = result.x
pd.DataFrame(θ_mle).to_csv('params_classrooms_2022_2024_hour_month_withAR1_ZScore.csv')
pd.DataFrame(result).to_csv('results_AR_messages.csv')

# Calculate AIC
AIC_AR1 = -2*(elec_loglik(θ_mle, X, y) - len(θ_mle))
