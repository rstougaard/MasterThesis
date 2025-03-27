from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shlex
from iminuit.cost import LeastSquares
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages
import pickle

with open("all_results_none_31_logpar_no_sys_error.pkl", "rb") as file:
    all_results_none = pickle.load(file)

focus_source = all_results_none["4FGL J0309.4-4000"]["No_Filtering"]

print(focus_source[:2])