import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings as warn
from warnings import filterwarnings
filterwarnings("ignore")

#read the data 
df = pd.read_csv(r"D:\Minor_python\data.csv")

data.head()