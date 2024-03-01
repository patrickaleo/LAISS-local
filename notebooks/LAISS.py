# import necessary packages
import time
import datetime

import math
import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import annoy
from annoy import AnnoyIndex
import random

import antares_client
from alerce.core import Alerce
alerce = Alerce()

import requests
from requests.auth import HTTPBasicAuth

import matplotlib
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import astropy.table as at
from astropy.table import MaskedColumn
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u
from astropy.visualization import PercentileInterval, AsinhStretch
from astroquery.mast import Catalogs
from astroquery.sdss import SDSS
from astroquery.simbad import Simbad

import light_curve as lc
from itertools import chain

from PIL import Image
import os
import sys
import shutil
import glob
import json

import astro_ghost
from astro_ghost.ghostHelperFunctions import getTransientHosts, getGHOST
import tempfile

# Throw RA/DEC into ghost with just DLR method, gentle starcut
# Sets environ var to find ghost.csv
os.environ['GHOST_PATH'] = './host_info'
# Then don't use getGHOST(real=True, verbose=verbose)
getGHOST(real=True,verbose=False)

import warnings
warnings.filterwarnings("ignore")

plt.style.use('fig_publication.mplstyle')

import argparse
parser = argparse.ArgumentParser(description='Run LAISS AD. Ex: python3 LAISS.py A B C D E F')

### EXAMPLE ###

# LAISS(l_or_ztfid_ref="ZTF18abydmfv",
#       lc_and_host_features=lc_and_host_features,
#       n=8,
#       use_lc_for_ann_only_bool=True, # currently doesn't work with YSE_snana_format or ANT IDs
#       use_ysepz_phot_snana_file=False,
#       show_lightcurves_grid=False,
#       show_hosts_grid=False,
#       run_AD_model=False,
#       savetables=False,
#       savefigs=False)

# Add command-line arguments for input, new data, and output file paths
parser.add_argument('l_or_ztfid_ref', help='l_or_ztfid_ref', nargs='?', type=str, const=1, default="ZTF21aaublej")
parser.add_argument('num_ann', help='num_ann', nargs='?', type=int, const=1, default=8)
parser.add_argument('use_lc_for_ann_only_bool', help='use_lc_for_ann_only_bool', nargs='?', type=bool, const=1, default=False)
parser.add_argument('use_ysepz_phot_snana_file', help='use_ysepz_phot_snana_file', nargs='?', type=bool, const=1, default=False)
parser.add_argument('show_lightcurves_grid', help='show_lightcurves_grid', nargs='?', type=bool, const=1, default=True)
parser.add_argument('show_hosts_grid', help='show_hosts_grid', nargs='?', type=bool, const=1, default=False) #TODO: currently pdf so doesn't show
parser.add_argument('run_AD_model', help='run_AD_model', nargs='?', type=bool, const=1, default=True)
parser.add_argument('savetables', help='savetables', nargs='?', type=bool, const=1, default=False)
parser.add_argument('savefigs', help='savefigs', nargs='?', type=bool, const=1, default=False)
args = parser.parse_args()

print(f"\nYou selected...\n\n LAISS(l_or_ztfid_ref={args.l_or_ztfid_ref}, \n lc_and_host_features=lc_and_host_features, \n n={args.num_ann},\
      \n use_lc_for_ann_only_bool={args.use_lc_for_ann_only_bool}, \n use_ysepz_phot_snana_file={args.use_ysepz_phot_snana_file},\
      \n show_lightcurves_grid={args.show_lightcurves_grid}, \n show_hosts_grid={args.show_hosts_grid}, \n run_AD_model={args.run_AD_model},\
      \n savetables={args.savetables}, \n savefigs={args.savefigs})")
print("\nRunning LAISS...\n")



####################

# GHOST getTransientHosts function with timeout
from timeout_decorator import timeout, TimeoutError


@timeout(120)  # Set a timeout of 60 seconds to query GHOST throughput APIs for host galaxy data
def getTransientHosts_with_timeout(**args):
    # time.sleep(6) #- to test
    return astro_ghost.ghostHelperFunctions.getTransientHosts(**args)


# Functions to extract light-curve features
def replace_magn_with_flux(s):
    if 'magnitude' in s:
        return s.replace('magnitudes', 'fluxes').replace('magnitude', 'flux')
    return f'{s} for flux light curve'


def create_base_features_class(
        magn_extractor,
        flux_extractor,
        bands=('R', 'g',),
):
    feature_names = ([f'{name}_magn' for name in magn_extractor.names]
                     + [f'{name}_flux' for name in flux_extractor.names])

    property_names = {band: [f'feature_{name}_{band}'.lower()
                             for name in feature_names]
                      for band in bands}

    features_count = len(feature_names)

    return feature_names, property_names, features_count


MAGN_EXTRACTOR = lc.Extractor(
    lc.Amplitude(),
    lc.AndersonDarlingNormal(),
    lc.BeyondNStd(1.0),
    lc.BeyondNStd(2.0),
    lc.Cusum(),
    lc.EtaE(),
    lc.InterPercentileRange(0.02),
    lc.InterPercentileRange(0.1),
    lc.InterPercentileRange(0.25),
    lc.Kurtosis(),
    lc.LinearFit(),
    lc.LinearTrend(),
    lc.MagnitudePercentageRatio(0.4, 0.05),
    lc.MagnitudePercentageRatio(0.2, 0.05),
    lc.MaximumSlope(),
    lc.Mean(),
    lc.MedianAbsoluteDeviation(),
    lc.PercentAmplitude(),
    lc.PercentDifferenceMagnitudePercentile(0.05),
    lc.PercentDifferenceMagnitudePercentile(0.1),
    lc.MedianBufferRangePercentage(0.1),
    lc.MedianBufferRangePercentage(0.2),
    lc.Periodogram(
        peaks=5,
        resolution=10.0,
        max_freq_factor=2.0,
        nyquist='average',
        fast=True,
        features=(
            lc.Amplitude(),
            lc.BeyondNStd(2.0),
            lc.BeyondNStd(3.0),
            lc.StandardDeviation(),
        ),
    ),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StandardDeviation(),
    lc.StetsonK(),
    lc.WeightedMean(),
)

FLUX_EXTRACTOR = lc.Extractor(
    lc.AndersonDarlingNormal(),
    lc.Cusum(),
    lc.EtaE(),
    lc.ExcessVariance(),
    lc.Kurtosis(),
    lc.MeanVariance(),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StetsonK(),
)


def remove_simultaneous_alerts(table):
    """Remove alert duplicates"""
    dt = np.diff(table['ant_mjd'], append=np.inf)
    return table[dt != 0]


def get_detections(photometry, band):
    """Extract clean light curve in given band from locus photometry"""
    band_lc = photometry[(photometry['ant_passband'] == band) & (~photometry['ant_mag'].isna())]
    idx = ~MaskedColumn(band_lc['ant_mag']).mask
    detections = remove_simultaneous_alerts(band_lc[idx])
    return detections


# From 106 available features from Kostya's lc_feature_extractor, use the 82 from SNAD Miner paper
# R and g bands
feature_names_r_g = ['feature_amplitude_magn_r',
                 'feature_anderson_darling_normal_magn_r',
                 'feature_beyond_1_std_magn_r',
                 'feature_beyond_2_std_magn_r',
                 'feature_cusum_magn_r',
                 #'feature_eta_e_magn_r',
                 'feature_inter_percentile_range_2_magn_r',
                 'feature_inter_percentile_range_10_magn_r',
                 'feature_inter_percentile_range_25_magn_r',
                 'feature_kurtosis_magn_r',
                 'feature_linear_fit_slope_magn_r',
                 'feature_linear_fit_slope_sigma_magn_r',
                 #'feature_linear_fit_reduced_chi2_magn_r',
                 #'feature_linear_trend_magn_r', # cadence removal
                 #'feature_linear_trend_sigma_magn_r',  # cadence removal
                 'feature_magnitude_percentage_ratio_40_5_magn_r',
                 'feature_magnitude_percentage_ratio_20_5_magn_r',
                 #'feature_maximum_slope_magn_r',
                 'feature_mean_magn_r',
                 'feature_median_absolute_deviation_magn_r',
                 'feature_percent_amplitude_magn_r',
                 'feature_median_buffer_range_percentage_10_magn_r',
                 'feature_median_buffer_range_percentage_20_magn_r',
                 'feature_percent_difference_magnitude_percentile_5_magn_r',
                 'feature_percent_difference_magnitude_percentile_10_magn_r',
                 #'feature_period_0_magn_r',  # should be negated
                 #'feature_period_s_to_n_0_magn_r', # cadence removal
                 #'feature_period_1_magn_r',
                 #'feature_period_s_to_n_1_magn_r', # cadence removal
                 #'feature_period_2_magn_r',
                 #'feature_period_s_to_n_2_magn_r', # cadence removal
                 #'feature_period_3_magn_r',
                 #'feature_period_s_to_n_3_magn_r', # cadence removal
                 #'feature_period_4_magn_r',
                 #'feature_period_s_to_n_4_magn_r', # cadence removal
                 #'feature_periodogram_amplitude_magn_r',
                 #'feature_periodogram_beyond_2_std_magn_r',  # cadence removal
                 #'feature_periodogram_beyond_3_std_magn_r',  # cadence removal
                 #'feature_periodogram_standard_deviation_magn_r',   # cadence removal
                 #'feature_chi2_magn_r',
                 'feature_skew_magn_r',
                 'feature_standard_deviation_magn_r',
                 'feature_stetson_k_magn_r',
                 'feature_weighted_mean_magn_r',
                 'feature_anderson_darling_normal_flux_r',
                 'feature_cusum_flux_r',
                 #'feature_eta_e_flux_r',
                 'feature_excess_variance_flux_r',
                 'feature_kurtosis_flux_r',
                 'feature_mean_variance_flux_r',
                 #'feature_chi2_flux_r',
                 'feature_skew_flux_r',
                 'feature_stetson_k_flux_r',

                 'feature_amplitude_magn_g',
                 'feature_anderson_darling_normal_magn_g',
                 'feature_beyond_1_std_magn_g',
                 'feature_beyond_2_std_magn_g',
                 'feature_cusum_magn_g',
                 #'feature_eta_e_magn_g',
                 'feature_inter_percentile_range_2_magn_g',
                 'feature_inter_percentile_range_10_magn_g',
                 'feature_inter_percentile_range_25_magn_g',
                 'feature_kurtosis_magn_g',
                 'feature_linear_fit_slope_magn_g',
                 'feature_linear_fit_slope_sigma_magn_g',
                 #'feature_linear_fit_reduced_chi2_magn_g',
                 #'feature_linear_trend_magn_g', # cadence removal
                 #'feature_linear_trend_sigma_magn_g',  # cadence removal
                 'feature_magnitude_percentage_ratio_40_5_magn_g',
                 'feature_magnitude_percentage_ratio_20_5_magn_g',
                 #'feature_maximum_slope_magn_g',
                 'feature_mean_magn_g',
                 'feature_median_absolute_deviation_magn_g',
                 'feature_median_buffer_range_percentage_10_magn_g',
                 'feature_median_buffer_range_percentage_20_magn_g',
                 'feature_percent_amplitude_magn_g',
                 'feature_percent_difference_magnitude_percentile_5_magn_g',
                 'feature_percent_difference_magnitude_percentile_10_magn_g',
                 #'feature_period_0_magn_g',  # should be negated
                 #'feature_period_s_to_n_0_magn_g', # cadence removal
                 #'feature_period_1_magn_g',
                 #'feature_period_s_to_n_1_magn_g', # cadence removal
                 #'feature_period_2_magn_g',
                 #'feature_period_s_to_n_2_magn_g', # cadence removal
                 #'feature_period_3_magn_g',
                 #'feature_period_s_to_n_3_magn_g', # cadence removal
                 #'feature_period_4_magn_g',
                 #'feature_period_s_to_n_4_magn_g', # cadence removal
                 #'feature_periodogram_amplitude_magn_g',
                 #'feature_periodogram_beyond_2_std_magn_g',  # cadence removal
                 #'feature_periodogram_beyond_3_std_magn_g', # cadence removal
                 #'feature_periodogram_standard_deviation_magn_g',  # cadence removal
                 #'feature_chi2_magn_g',
                 'feature_skew_magn_g',
                 'feature_standard_deviation_magn_g',
                 'feature_stetson_k_magn_g',
                 'feature_weighted_mean_magn_g',
                 'feature_anderson_darling_normal_flux_g',
                 'feature_cusum_flux_g',
                 #'feature_eta_e_flux_g',
                 'feature_excess_variance_flux_g',
                 'feature_kurtosis_flux_g',
                 'feature_mean_variance_flux_g',
                 #'feature_chi2_flux_g',
                 'feature_skew_flux_g',
                 'feature_stetson_k_flux_g']

feature_names_hostgal = [
    #  'Unnamed: 0',
    #  'level_0',
    #  'index',
    #  'objName',
    #  'objAltName1',
    #  'objAltName2',
    #  'objAltName3',
    #  'objID',
    #  'uniquePspsOBid',
    #  'ippObjID',
    #  'surveyID',
    #  'htmID',
    #  'zoneID',
    #  'tessID',
    #  'projectionID',
    #  'skyCellID',
    #  'randomID',
    #  'batchID',
    #  'dvoRegionID',
    #  'processingVersion',
    #  'objInfoFlag',
    #  'qualityFlag',
    #  'raStack',
    #  'decStack',
    #  'raStackErr',
    #  'decStackErr',
    #  'raMean',
    #  'decMean',
    #  'raMeanErr',
    #  'decMeanErr',
    #  'epochMean',
    #  'posMeanChisq',
    #  'cx',
    #  'cy',
    #  'cz',
    #  'lambda',
    #  'beta',
    #  'l',
    #  'b',
    #  'nStackObjectRows',
    #  'nStackDetections',
    #  'nDetections',
    #  'ng',
    #  'nr',
    #  'ni',
    #  'nz',
    #  'ny',
    #  'uniquePspsSTid',
    #  'primaryDetection',
    #  'bestDetection',
    #  'gippDetectID',
    #  'gstackDetectID',
    #  'gstackImageID',
    #  'gra',
    #  'gdec',
    #  'graErr',
    #  'gdecErr',
    #  'gEpoch',
    #  'gPSFMag',
    #  'gPSFMagErr',
    #  'gApMag',
    #  'gApMagErr',
    #  'gKronMag',
    #  'gKronMagErr',
    #  'ginfoFlag',
    #  'ginfoFlag2',
    #  'ginfoFlag3',
    #  'gnFrames',
    #  'gxPos',
    #  'gyPos',
    #  'gxPosErr',
    #  'gyPosErr',
    #  'gpsfMajorFWHM',
    #  'gpsfMinorFWHM',
    #  'gpsfTheta',
    #  'gpsfCore',
    #  'gpsfLikelihood',
    #  'gpsfQf',
    #  'gpsfQfPerfect',
    #  'gpsfChiSq',
     'gmomentXX',
     'gmomentXY',
     'gmomentYY',
     'gmomentR1',
     'gmomentRH',
     'gPSFFlux',
    #  'gPSFFluxErr',
     'gApFlux',
    #  'gApFluxErr',
    #  'gApFillFac',
    #  'gApRadius',
     'gKronFlux',
    #  'gKronFluxErr',
     'gKronRad',
    #  'gexpTime',
     'gExtNSigma',
    #  'gsky',
    #  'gskyErr',
    #  'gzp',
    #  'gPlateScale',
    #  'rippDetectID',
    #  'rstackDetectID',
    #  'rstackImageID',
    #  'rra',
    #  'rdec',
    #  'rraErr',
    #  'rdecErr',
    #  'rEpoch',
    # 'rPSFMag',
    #  'rPSFMagErr',
    # 'rApMag',
    #  'rApMagErr',
    # 'rKronMag',
    #  'rKronMagErr',
    #  'rinfoFlag',
    #  'rinfoFlag2',
    #  'rinfoFlag3',
    #  'rnFrames',
    #  'rxPos',
    #  'ryPos',
    #  'rxPosErr',
    #  'ryPosErr',
    #  'rpsfMajorFWHM',
    #  'rpsfMinorFWHM',
    #  'rpsfTheta',
    #  'rpsfCore',
    #  'rpsfLikelihood',
    #  'rpsfQf',
    #  'rpsfQfPerfect',
    #  'rpsfChiSq',
     'rmomentXX',
     'rmomentXY',
     'rmomentYY',
     'rmomentR1',
     'rmomentRH',
    'rPSFFlux',
    #  'rPSFFluxErr',
    'rApFlux',
    #  'rApFluxErr',
    #  'rApFillFac',
    # 'rApRadius',
    'rKronFlux',
    #  'rKronFluxErr',
    'rKronRad',
    #  'rexpTime',
     'rExtNSigma',
    #  'rsky',
    #  'rskyErr',
    #  'rzp',
    #  'rPlateScale',
    #  'iippDetectID',
    #  'istackDetectID',
    #  'istackImageID',
    #  'ira',
    #  'idec',
    #  'iraErr',
    #  'idecErr',
    #  'iEpoch',
    #  'iPSFMag',
    #  'iPSFMagErr',
    #  'iApMag',
    #  'iApMagErr',
    #  'iKronMag',
    #  'iKronMagErr',
    #  'iinfoFlag',
    #  'iinfoFlag2',
    #  'iinfoFlag3',
    #  'inFrames',
    #  'ixPos',
    #  'iyPos',
    #  'ixPosErr',
    #  'iyPosErr',
    #  'ipsfMajorFWHM',
    #  'ipsfMinorFWHM',
    #  'ipsfTheta',
    #  'ipsfCore',
    #  'ipsfLikelihood',
    #  'ipsfQf',
    #  'ipsfQfPerfect',
    #  'ipsfChiSq',
      'imomentXX',
     'imomentXY',
     'imomentYY',
     'imomentR1',
     'imomentRH',
     'iPSFFlux',
    #  'iPSFFluxErr',
     'iApFlux',
    #  'iApFluxErr',
    #  'iApFillFac',
    #  'iApRadius',
     'iKronFlux',
    #  'iKronFluxErr',
     'iKronRad',
    #  'iexpTime',
      'iExtNSigma',
    #  'isky',
    #  'iskyErr',
    #  'izp',
    #  'iPlateScale',
    #  'zippDetectID',
    #  'zstackDetectID',
    #  'zstackImageID',
    #  'zra',
    #  'zdec',
    #  'zraErr',
    #  'zdecErr',
    #  'zEpoch',
    #  'zPSFMag',
    #  'zPSFMagErr',
    #  'zApMag',
    #  'zApMagErr',
    #  'zKronMag',
    #  'zKronMagErr',
    #  'zinfoFlag',
    #  'zinfoFlag2',
    #  'zinfoFlag3',
    #  'znFrames',
    #  'zxPos',
    #  'zyPos',
    #  'zxPosErr',
    #  'zyPosErr',
    #  'zpsfMajorFWHM',
    #  'zpsfMinorFWHM',
    #  'zpsfTheta',
    #  'zpsfCore',
    #  'zpsfLikelihood',
    #  'zpsfQf',
    #  'zpsfQfPerfect',
    #  'zpsfChiSq',
      'zmomentXX',
     'zmomentXY',
     'zmomentYY',
     'zmomentR1',
     'zmomentRH',
     'zPSFFlux',
    # #  'zPSFFluxErr',
     'zApFlux',
    # #  'zApFluxErr',
    # #  'zApFillFac',
    # #  'zApRadius',
     'zKronFlux',
    # #  'zKronFluxErr',
     'zKronRad',
    # #  'zexpTime',
      'zExtNSigma',
    #  'zsky',
    #  'zskyErr',
    #  'zzp',
    #  'zPlateScale',
    #  'yippDetectID',
    #  'ystackDetectID',
    #  'ystackImageID',
    #  'yra',
    #  'ydec',
    #  'yraErr',
    #  'ydecErr',
    #  'yEpoch',
    #  'yPSFMag',
    #  'yPSFMagErr',
    #  'yApMag',
    #  'yApMagErr',
    #  'yKronMag',
    #  'yKronMagErr',
    #  'yinfoFlag',
    #  'yinfoFlag2',
    #  'yinfoFlag3',
    #  'ynFrames',
    #  'yxPos',
    #  'yyPos',
    #  'yxPosErr',
    #  'yyPosErr',
    #  'ypsfMajorFWHM',
    #  'ypsfMinorFWHM',
    #  'ypsfTheta',
    #  'ypsfCore',
    #  'ypsfLikelihood',
    #  'ypsfQf',
    #  'ypsfQfPerfect',
    #  'ypsfChiSq',
      'ymomentXX',
      'ymomentXY',
      'ymomentYY',
      'ymomentR1',
      'ymomentRH',
      'yPSFFlux',
    # #   'yPSFFluxErr',
      'yApFlux',
    # #   'yApFluxErr',
    # #   'yApFillFac',
    # #  'yApRadius',
     'yKronFlux',
    # #  'yKronFluxErr',
     'yKronRad',
    # #  'yexpTime',
      'yExtNSigma',
    #  'ysky',
    #  'yskyErr',
    #  'yzp',
    #  'yPlateScale',
    #  'distance',
    #  'SkyMapper_StarClass',
    #  'gelong',
    #  'g_a',
    #  'g_b',
    #  'g_pa',
    #  'relong',
    #  'r_a',
    #  'r_b',
    #  'r_pa',
    #  'ielong',
    #  'i_a',
    #  'i_b',
    #  'i_pa',
    #  'zelong',
    #  'z_a',
    #  'z_b',
    #  'z_pa',
       'i-z', # try throwing in
    #    'g-r',
    #    'r-i',
    #    'g-i',
    #    'z-y',
    #   'g-rErr',
    #   'r-iErr',
    #   'i-zErr',
    #   'z-yErr',
     'gApMag_gKronMag',
     'rApMag_rKronMag',
     'iApMag_iKronMag',
     'zApMag_zKronMag',
     'yApMag_yKronMag',
     '7DCD',
    #  'NED_name',
    #  'NED_type',
    #  'NED_vel',
    #  'NED_redshift',
    #  'NED_mag',
    #  'class',
       'dist/DLR',
    #   'dist',
    #  'TransientClass',
    #  'TransientRA',
    #  'TransientDEC'
       ]


lc_and_host_features = feature_names_r_g + feature_names_hostgal

# Hyperparameters for best AD model
n_estimators = 100
max_depth = 35
random_state = 11
max_features = 35  # {“sqrt”, “log2”, None}, int or float, default=”sqrt” - sqrt(120) ~ 10
class_weight = {"Normal": 1, "Other": 1}  # "balanced"

figure_path = f"/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/RFC/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/figures"
model_path = f"/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/RFC/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/model"
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

with open(
        f'{model_path}/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced.pkl',
        'rb') as f:
    clf = pickle.load(f)


def plot_RFC_prob_vs_lc_ztfid(clf, anom_ztfid, anom_spec_cls, anom_spec_z, anom_thresh, lc_and_hosts_df,
                              lc_and_hosts_df_120d, ref_info, savefig, figure_path):
    anom_thresh = anom_thresh
    anom_obj_df = lc_and_hosts_df_120d

    try:
        pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
        pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
        pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
        num_anom_epochs = len(np.where(pred_prob_anom[:, 1] >= anom_thresh)[0])
    except:
        print(f"{anom_ztfid} has some NaN host galaxy values from PS1 catalog. Skip!")
        return

    try:
        anom_idx = lc_and_hosts_df.iloc[np.where(pred_prob_anom[:, 1] >= anom_thresh)[0][0]].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {anom_ztfid}.")
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs)

    ztf_id_ref = anom_ztfid

    ref_info = ref_info

    df_ref = ref_info.timeseries.to_pandas()

    df_ref_g = df_ref[(df_ref.ant_passband == 'g') & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == 'R') & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[['ant_mag']].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[['ant_mag']].reset_index().idxmin().ant_mag

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 6.5))
    ax1.invert_yaxis()
    ax1.errorbar(x=df_ref_r.ant_mjd, y=df_ref_r.ant_mag, yerr=df_ref_r.ant_magerr, fmt='o', c='r', label=r'ZTF-$r$')
    ax1.errorbar(x=df_ref_g.ant_mjd, y=df_ref_g.ant_mag, yerr=df_ref_g.ant_magerr, fmt='o', c='g', label=r'ZTF-$g$')
    if anom_idx_is == True:
        ax1.axvline(x=lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0],
                    label="Tag anomalous", color='dodgerblue', ls='--')
        # ax1.axvline(x=59323, label="Orig. spectrum", color='darkviolet', ls='-.')
        mjd_cross_thresh = round(lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0], 3)

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left) / (right - left)
        # mjd_anom_per2 = (59323 - left)/(right - left)
        plt.text(mjd_anom_per + 0.073, -0.075, f"t$_a$ = {int(mjd_cross_thresh)}", horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize=16, color='dodgerblue')
        # plt.text(mjd_anom_per2+0.12, 0.035, f"t$_s$ = {int(59323)}", horizontalalignment='center',
        # verticalalignment='center', transform=ax1.transAxes, fontsize=16, color='darkviolet')
        print("MJD crossed thresh:", mjd_cross_thresh)

    print(f'https://alerce.online/object/{anom_ztfid}')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 0], label=r'$p(Normal)$')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 1], label=r'$p(Anomaly)$')

    if anom_spec_z is None:
        anom_spec_z = "None"
    elif isinstance(anom_spec_z, float):
        anom_spec_z = round(anom_spec_z, 3)
    else:
        anom_spec_z = anom_spec_z
    ax1.set_title(fr"{anom_ztfid} ({anom_spec_cls}, $z$={anom_spec_z})", pad=25, fontsize=14) #tweaking fontsize, bbox_to_anchor for display
    plt.xlabel('MJD', fontsize=14)
    ax1.set_ylabel('Magnitude', fontsize=14)
    ax2.set_ylabel('Probability (%)', fontsize=14)

    if anom_idx_is == True:
        ax1.legend(loc='upper right', ncol=3, bbox_to_anchor=(1.0, 1.15), frameon=False, fontsize=12)
    # if anom_idx_is == True: ax1.legend(loc='upper right', ncol=4, bbox_to_anchor=(1.05,1.12), columnspacing=0.65, frameon=False, fontsize=14)
    else:
        ax1.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.75, 1.15), frameon=False, fontsize=12)
    ax2.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.87, 1.15), frameon=False, fontsize=12)

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        plt.savefig(f"{figure_path}/{anom_ztfid}_AD_run_timeseries.pdf", dpi=300, bbox_inches='tight')

    plt.show()


def plot_RFC_prob_vs_lc_yse_IAUid(clf, IAU_name, anom_ztfid, anom_spec_cls, anom_spec_z, anom_thresh, lc_and_hosts_df,
                                  lc_and_hosts_df_120d, yse_lightcurve, savefig, figure_path):
    anom_thresh = anom_thresh
    anom_obj_df = lc_and_hosts_df_120d

    try:
        pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
        pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
        pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
        num_anom_epochs = len(np.where(pred_prob_anom[:, 1] >= anom_thresh)[0])
    except:
        print(f"{anom_ztfid} has some NaN host galaxy values from PS1 catalog. Skip!")
        return

    try:
        anom_idx = lc_and_hosts_df.iloc[np.where(pred_prob_anom[:, 1] >= anom_thresh)[0][0]].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {anom_ztfid}.")
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs)

    ztf_id_ref = anom_ztfid

    df_ref_g = yse_lightcurve[(yse_lightcurve.FLT == 'g')]
    df_ref_r = yse_lightcurve[(yse_lightcurve.FLT == 'R')]

    mjd_idx_at_min_mag_r_ref = df_ref_r[['MAG']].reset_index().idxmin().MAG
    mjd_idx_at_min_mag_g_ref = df_ref_g[['MAG']].reset_index().idxmin().MAG

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 10))
    ax1.invert_yaxis()

    df_ref_g_ztf = df_ref_g[df_ref_g.TELESCOPE == 'P48']
    df_ref_g_ps1 = df_ref_g[df_ref_g.TELESCOPE == 'Pan-STARRS1']
    df_ref_r_ztf = df_ref_r[df_ref_r.TELESCOPE == 'P48']
    df_ref_r_ps1 = df_ref_r[df_ref_r.TELESCOPE == 'Pan-STARRS1']

    ax1.errorbar(x=df_ref_r_ztf.MJD, y=df_ref_r_ztf.MAG, yerr=df_ref_r_ztf.MAGERR, fmt='o', c='r', label=r'ZTF-$r$')
    ax1.errorbar(x=df_ref_g_ztf.MJD, y=df_ref_g_ztf.MAG, yerr=df_ref_g_ztf.MAGERR, fmt='o', c='g', label=r'ZTF-$g$')
    ax1.errorbar(x=df_ref_r_ps1.MJD, y=df_ref_r_ps1.MAG, yerr=df_ref_r_ps1.MAGERR, fmt='s', c='r', label=r'PS1-$r$')
    ax1.errorbar(x=df_ref_g_ps1.MJD, y=df_ref_g_ps1.MAG, yerr=df_ref_g_ps1.MAGERR, fmt='s', c='g', label=r'PS1-$g$')

    if anom_idx_is == True:
        ax1.axvline(x=lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0],
                    label="Tagged anomalous", color='darkblue', ls='--')
        mjd_cross_thresh = round(lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0], 3)

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left) / (right - left)
        plt.text(mjd_anom_per + 0.073, -0.075, f"t = {int(mjd_cross_thresh)}", horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize=16)
        print("MJD crossed thresh:", mjd_cross_thresh)

    print(f'https://ziggy.ucolick.org/yse/transient_detail/{IAU_name}/')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 0], label=r'$p(Normal)$')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 1], label=r'$p(Anomaly)$')

    if anom_spec_z is None:
        anom_spec_z = "None"
    elif isinstance(anom_spec_z, float):
        anom_spec_z = round(anom_spec_z, 3)
    else:
        anom_spec_z = anom_spec_z
    ax1.set_title(fr"{anom_ztfid} ({anom_spec_cls}, $z$={anom_spec_z})", pad=25)
    plt.xlabel('MJD')
    ax1.set_ylabel('Magnitude')
    ax2.set_ylabel('Probability (%)')

    if anom_idx_is == True:
        ax1.legend(loc='upper right', ncol=5, bbox_to_anchor=(1.1, 1.12), columnspacing=0.45, frameon=False,
                   fontsize=14)
    else:
        ax1.legend(loc='upper right', ncol=4, bbox_to_anchor=(1.03, 1.12), frameon=False, fontsize=14)
    ax2.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.87, 1.12), frameon=False, fontsize=14)

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        plt.savefig(f"{figure_path}/{anom_ztfid}_AD_run_timeseries.pdf", dpi=300, bbox_inches='tight')

    plt.show()


# with open("./loci.pkl", "rb") as f:
#     loci = pickle.load(f)

# loci_df = pd.DataFrame.from_dict(loci[0:1000000])
# loci_df = pd.concat([loci_df.drop(['properties'], axis=1), loci_df['properties'].apply(pd.Series)], axis=1)

# coord_testset = SkyCoord(list(loci_df['ra']),
#                          list(loci_df['dec']),
#                          frame="icrs",
#                          unit='deg')
# galcoord_testset = coord_testset.galactic
# loci_df['locus_gal_l'] = galcoord_testset.l.degree
# loci_df['locus_gal_b'] = galcoord_testset.b.degree
# loci_df.drop('replaced_by', axis=1, inplace=True)
# loci_df.drop('htm16', axis=1, inplace=True)
# loci_df.drop('watch_list_ids', axis=1, inplace=True)
# loci_df.drop('watch_object_ids', axis=1, inplace=True)

# # cut on anything in galactic plane
# hard_mask = (np.abs(loci_df['locus_gal_b']) < 15)
# loci_df = loci_df[~hard_mask]


# # cut on anything that's a likely star (in a star catalog), or AGN
# # 'bright_guide_star_cat' is sometimes a galaxy, so keep
# drop_values = ['sdss_stars', 'asassn_variable_catalog', 'asassn_variable_catalog_v2_20190802', 'veron_agn_qso']
# loci_df = loci_df[~loci_df['catalogs'].apply(lambda x: any(item in x for item in drop_values))]

# loci_df.to_csv(f'./loci_df_{len(loci_df)}objects_cut_stars_and_gal_plane.csv', compression='gzip')


bigbank_df = pd.read_csv(f'./loci_df_271688objects_cut_stars_and_gal_plane.csv', compression='gzip')
bigbank_df = bigbank_df.set_index('ztf_object_id')
bigbank_df = bigbank_df[feature_names_r_g]
bigbank_df = bigbank_df.dropna()
bigbank_df # 90k data set

# LC features only annoy index, no PCA
feat_arr = np.array(bigbank_df[feature_names_r_g])
idx_arr = np.array(bigbank_df[feature_names_r_g].index)

ntrees = 100
# Create or load the ANNOY index
index_nm = f"./bigbank_90k_LCfeats_only_annoy_index_{ntrees}trees"
# Save the index array to a binary file
np.save(f"{index_nm}_idx_arr.npy", idx_arr)

index_file = f"./bigbank_90k_LCfeats_only_annoy_index_{ntrees}trees.ann"  # Choose a filename
index_dim = feat_arr.shape[1]  # Dimension of the index

# Check if the index file exists
if not os.path.exists(index_file):
    print("Saving new ANNOY index")
    # If the index file doesn't exist, create and build the index
    index = annoy.AnnoyIndex(index_dim, metric='manhattan')

    # Add items to the index
    for i in range(len(idx_arr)):
        index.add_item(i, feat_arr[i])

    # Build the index
    index.build(ntrees)

    # Save the index to a file
    index.save(index_file)
else:
    print("Loading previously saved ANNOY LC-only index")
    # If the index file exists, load it
    index = annoy.AnnoyIndex(index_dim, metric='manhattan')
    index.load(index_file)
    idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)


# for now, just use original dataset bank

# dataset_bank = pd.read_csv('../loci_dbs/alerce_cut/dataset_bank.csv.gz', compression='gzip', index_col=0)

# # from ps1_psc.ipynb
# # used mask (keep_df.pgal_host < 0.5) & (keep_df.pgal_transient < 0.5)
# keep_df = pd.read_csv(f'../loci_dbs/alerce_cut/keep_df.csv.gz', compression='gzip')
# keep_df = keep_df.drop_duplicates(subset='ztf_object_id', keep='first') # keep first occurance of ztfid
# keep_df = keep_df.set_index('ztf_object_id')

# # merge df1 with df2 on the index and add the 'stamp_cls' column
# keep_df = keep_df.merge(dataset_bank[['stamp_cls']], left_index=True, right_index=True, suffixes=('', '_cls'))

# dataset_bank_orig = dataset_bank[dataset_bank.index.isin(keep_df.index)]
# dataset_bank_orig = dataset_bank_orig[lc_and_host_features]

# dataset_label_orig = list(dataset_bank_orig.index)

# Load the spec & phot dataset_bank used for train/test (before upsampling w/ SMOTE)
# All real events
dataset_bank_orig = pd.read_csv('../loci_dbs/alerce_cut/dataset_bank_orig_5472objs.csv.gz', compression='gzip', index_col=0)
dataset_bank = pd.read_csv('../loci_dbs/alerce_cut/dataset_bank.csv.gz', compression='gzip', index_col=0)
#dataset_bank_orig_w_hosts_ra_dec = dataset_bank[dataset_bank.index.isin(dataset_bank_orig.index)]
#dataset_bank_orig_w_hosts_ra_dec.to_csv('../loci_dbs/alerce_cut/dataset_bank_orig_w_hosts_ra_dec_5472objs.csv.gz', compression='gzip')
dataset_bank_orig_w_hosts_ra_dec = pd.read_csv('../loci_dbs/alerce_cut/dataset_bank_orig_w_hosts_ra_dec_5472objs.csv.gz', compression='gzip', index_col=0)

# LC features only annoy index, no PCA
feat_arr = np.array(dataset_bank_orig[feature_names_r_g])
idx_arr = np.array(dataset_bank_orig[feature_names_r_g].index)

# Create or load the ANNOY index
index_nm = f"./dataset_bank_LCfeats_only_annoy_index"
# Save the index array to a binary file
np.save(f"{index_nm}_idx_arr.npy", idx_arr)

# Create or load the ANNOY index
index_file = "./dataset_bank_LCfeats_only_annoy_index.ann"  # Choose a filename
index_dim = feat_arr.shape[1]  # Dimension of the index

# Check if the index file exists
if not os.path.exists(index_file):
    print("Saving new ANNOY index")
    # If the index file doesn't exist, create and build the index
    index = annoy.AnnoyIndex(index_dim, metric='manhattan')

    # Add items to the index
    for i in range(len(idx_arr)):
        index.add_item(i, feat_arr[i])

    # Build the index
    index.build(1000)  # 1000 trees

    # Save the index to a file
    index.save(index_file)
else:
    print("Loading previously saved ANNOY LC-only index")
    # If the index file exists, load it
    index = annoy.AnnoyIndex(index_dim, metric='manhattan')
    index.load(index_file)
    idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

# LC + host features annoy index, w/ PCA

feat_arr = np.array(dataset_bank_orig)
idx_arr = np.array(dataset_bank_orig.index)
scaler = preprocessing.StandardScaler()

# Set a random seed for PCA
random_seed = 42  # Choose your desired random seed

# Scale the features
feat_arr_scaled = scaler.fit_transform(feat_arr)

# Initialize PCA with 60 principal components
n_components = 60
pca = PCA(n_components=n_components, random_state=random_seed)

# Apply PCA
feat_arr_scaled_pca = pca.fit_transform(feat_arr_scaled)

# Print the explained variance
# print(np.cumsum(pca.explained_variance_ratio_))
# print(feat_arr_scaled_pca)

# Set a random seed for reproducibility
np.random.seed(random_seed)

# Create or load the ANNOY index
index_nm = f"./dataset_bank_60pca_annoy_index"
# Save the index array to a binary file
np.save(f"{index_nm}_idx_arr.npy", idx_arr)
np.save(f"{index_nm}_feat_arr.npy", feat_arr)
np.save(f"{index_nm}_feat_arr_scaled.npy", feat_arr_scaled)
np.save(f"{index_nm}_feat_arr_scaled_pca.npy", feat_arr_scaled_pca)

# Create or load the ANNOY index
index_file = "./dataset_bank_60pca_annoy_index.ann"  # Choose a filename
index_dim = feat_arr_scaled_pca.shape[1]  # Dimension of the index

# Check if the index file exists
if not os.path.exists(index_file):
    print("Saving new ANNOY index")
    # If the index file doesn't exist, create and build the index
    index = annoy.AnnoyIndex(index_dim, metric='manhattan')

    # Add items to the index
    for i in range(len(idx_arr)):
        index.add_item(i, feat_arr_scaled_pca[i])

    # Build the index
    index.build(1000)  # 1000 trees

    # Save the index to a file
    index.save(index_file)
else:
    print("Loading previously saved ANNOY index")
    # If the index file exists, load it
    index = annoy.AnnoyIndex(index_dim, metric='manhattan')
    index.load(index_file)
    idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)


def extract_lc_and_host_features(ztf_id_ref, use_lc_for_ann_only_bool, show_lc=False, show_host=True):
    start_time = time.time()
    ztf_id_ref = ztf_id_ref  # 'ZTF20aalxlis' #'ZTF21abmspzt'
    df_path = "/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries"

    try:
        ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id_ref)
        df_ref = ref_info.timeseries.to_pandas()
    except:
        print("antares_client can't find this object. Skip! Continue...")
        return

    df_ref_g = df_ref[(df_ref.ant_passband == 'g') & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == 'R') & (~df_ref.ant_mag.isna())]

    try:
        mjd_idx_at_min_mag_r_ref = df_ref_r[['ant_mag']].reset_index().idxmin().ant_mag
        mjd_idx_at_min_mag_g_ref = df_ref_g[['ant_mag']].reset_index().idxmin().ant_mag
    except:
        print(f"No obs for {ztf_id_ref}. pass!\n")
        return

    if show_lc:
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.gca().invert_yaxis()

        ax.errorbar(x=df_ref_r.ant_mjd, y=df_ref_r.ant_mag, yerr=df_ref_r.ant_magerr, fmt='o', c='r',
                    label=f'REF: {ztf_id_ref}')
        ax.errorbar(x=df_ref_g.ant_mjd, y=df_ref_g.ant_mag, yerr=df_ref_g.ant_magerr, fmt='o', c='g')
        plt.show()

    min_obs_count = 4

    lightcurve = ref_info.lightcurve
    # print("lightcurve", lightcurve)
    feature_names, property_names, features_count = create_base_features_class(MAGN_EXTRACTOR, FLUX_EXTRACTOR)

    g_obs = list(get_detections(lightcurve, 'g').ant_mjd.values)
    r_obs = list(get_detections(lightcurve, 'R').ant_mjd.values)
    mjd_l = sorted(g_obs + r_obs)

    lc_properties_d_l = []
    len_det_counter_r, len_det_counter_g = 0, 0

    band_lc = lightcurve[(~lightcurve['ant_mag'].isna())]
    idx = ~MaskedColumn(band_lc['ant_mag']).mask
    all_detections = remove_simultaneous_alerts(band_lc[idx])
    for ob, mjd in enumerate(mjd_l):  # requires 4 obs
        # do time evolution of detections - in chunks

        detections_pb = all_detections[all_detections['ant_mjd'].values <= mjd]
        # print(detections)
        lc_properties_d = {}
        for band, names in property_names.items():
            detections = detections_pb[detections_pb['ant_passband'] == band]

            # Ensure locus has >3 obs for calculation
            if (len(detections) < min_obs_count):
                continue
            # print(detections)

            t = detections['ant_mjd'].values
            m = detections['ant_mag'].values
            merr = detections['ant_magerr'].values
            flux = np.power(10.0, -0.4 * m)
            fluxerr = 0.5 * flux * (np.power(10.0, 0.4 * merr) - np.power(10.0, -0.4 * merr))

            magn_features = MAGN_EXTRACTOR(
                t,
                m,
                merr,
                fill_value=None,
            )
            flux_features = FLUX_EXTRACTOR(
                t,
                flux,
                fluxerr,
                fill_value=None,
            )

            # After successfully calculating features, set locus properties and tag
            lc_properties_d["obs_num"] = int(ob)
            lc_properties_d["mjd_cutoff"] = mjd
            lc_properties_d["ztf_object_id"] = ztf_id_ref
            # print(band, m)
            for name, value in zip(names, chain(magn_features, flux_features)):
                lc_properties_d[name] = value
                # if name == "feature_amplitude_magn_g": print(m, value, band)
            # print("%%%%%%%%")
        lc_properties_d_l.append(lc_properties_d)

    lc_properties_d_l = [d for d in lc_properties_d_l if d]
    lc_properties_df = pd.DataFrame(lc_properties_d_l)
    if len(lc_properties_df) == 0:
        print(f"Not enough obs for {ztf_id_ref}. pass!\n")
        return
    print(f"Extracted LC features for {ztf_id_ref}!")

    end_time = time.time()
    print(f"Extracted LC features in {end_time - start_time}s!")

    if not use_lc_for_ann_only_bool:

        # Get GHOST features
        ra, dec = np.mean(df_ref.ant_ra), np.mean(df_ref.ant_dec)
        snName = [ztf_id_ref, ztf_id_ref]
        snCoord = [SkyCoord(ra * u.deg, dec * u.deg, frame='icrs'), SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                hosts = getTransientHosts_with_timeout(transientName=snName, snCoord=snCoord, GLADE=True, verbose=0,
                                                       starcut='gentle', ascentMatch=False, savepath=tmp,
                                                       redo_search=False)
            except:
                print(f"GHOST error for {ztf_id_ref}. Retry without GLADE. \n")
                hosts = getTransientHosts_with_timeout(transientName=snName, snCoord=snCoord, GLADE=False, verbose=0,
                                                       starcut='gentle', ascentMatch=False, savepath=tmp,
                                                       redo_search=False)

        if len(hosts) > 1:
            hosts = pd.DataFrame(hosts.loc[0]).T

        hosts_df = hosts[feature_names_hostgal]
        hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]

        if len(hosts_df) < 1:
            # if any features are nan, we can't use as input
            print(f"Some features are NaN for {ztf_id_ref}. Skip!\n")
            return

        if show_host:
            print(
                f'http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color')

        hosts_df = hosts[feature_names_hostgal]
        hosts_df = pd.concat([hosts_df] * len(lc_properties_df), ignore_index=True)

        lc_and_hosts_df = pd.concat([lc_properties_df, hosts_df], axis=1)
        lc_and_hosts_df = lc_and_hosts_df.set_index('ztf_object_id')
        lc_and_hosts_df['raMean'] = hosts.raMean.values[0]
        lc_and_hosts_df['decMean'] = hosts.decMean.values[0]
        lc_and_hosts_df.to_csv(f'{df_path}/{lc_and_hosts_df.index[0]}_timeseries.csv')

    else:
        print("Saving for lc timeseries only")
        lc_properties_df = lc_properties_df.set_index('ztf_object_id')
        lc_properties_df.to_csv(f'{df_path}/{lc_properties_df.index[0]}_timeseries.csv')

    print(f"Saved results for {ztf_id_ref}!\n")


def extract_lc_and_host_features_YSE_snana_format(IAU_name, ztf_id_ref, yse_lightcurve, ra, dec, show_lc=False,
                                                  show_host=False):
    IAU_name = IAU_name
    df_path = "/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries"

    min_obs_count = 4

    lightcurve = yse_lightcurve
    feature_names, property_names, features_count = create_base_features_class(MAGN_EXTRACTOR, FLUX_EXTRACTOR)

    g_obs = list(yse_lightcurve[yse_lightcurve.FLT == "g"].MJD)
    r_obs = list(yse_lightcurve[yse_lightcurve.FLT == "R"].MJD)
    mjd_l = sorted(g_obs + r_obs)

    lc_properties_d_l = []
    len_det_counter_r, len_det_counter_g = 0, 0

    all_detections = yse_lightcurve
    for ob, mjd in enumerate(mjd_l):  # requires 4 obs
        # do time evolution of detections - in chunks
        detections_pb = all_detections[all_detections["MJD"].values <= mjd]
        # print(detections)
        lc_properties_d = {}
        for band, names in property_names.items():
            detections = detections_pb[detections_pb["FLT"] == band]

            # Ensure locus has >3 obs for calculation
            if (len(detections) < min_obs_count):
                continue
            # print(detections)

            t = detections['MJD'].values
            m = detections['MAG'].values
            merr = detections['MAGERR'].values
            flux = detections['FLUXCAL'].values
            fluxerr = detections['FLUXCALERR'].values

            try:
                magn_features = MAGN_EXTRACTOR(
                    t,
                    m,
                    merr,
                    fill_value=None,
                )
            except:
                print(f"{IAU_name} is maybe not sorted?")
                return

            flux_features = FLUX_EXTRACTOR(
                t,
                flux,
                fluxerr,
                fill_value=None,
            )

            # After successfully calculating features, set locus properties and tag
            lc_properties_d["obs_num"] = int(ob)
            lc_properties_d["mjd_cutoff"] = mjd
            lc_properties_d["ztf_object_id"] = ztf_id_ref
            # print(band, m)
            for name, value in zip(names, chain(magn_features, flux_features)):
                lc_properties_d[name] = value
                # if name == "feature_amplitude_magn_g": print(m, value, band)
            # print("%%%%%%%%")
        lc_properties_d_l.append(lc_properties_d)

    lc_properties_d_l = [d for d in lc_properties_d_l if d]
    lc_properties_df = pd.DataFrame(lc_properties_d_l)
    if len(lc_properties_df) == 0:
        print(f"Not enough obs for {IAU_name}. pass!\n")
        return
    print(f"Extracted LC features for {IAU_name}/{ztf_id_ref}!")

    # Get GHOST features
    ra, dec = float(ra), float(dec)
    snName = [IAU_name, IAU_name]
    snCoord = [SkyCoord(ra * u.deg, dec * u.deg, frame='icrs'), SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            hosts = getTransientHosts(transientName=snName, snCoord=snCoord, GLADE=True, verbose=0,
                                      starcut='gentle', ascentMatch=False, savepath=tmp, redo_search=False)
        except:
            print(f"GHOST error for {IAU_name}. Retry without GLADE. \n")
            hosts = getTransientHosts(transientName=snName, snCoord=snCoord, GLADE=False, verbose=0,
                                      starcut='gentle', ascentMatch=False, savepath=tmp, redo_search=False)

    if len(hosts) > 1:
        hosts = pd.DataFrame(hosts.loc[0]).T

    hosts_df = hosts[feature_names_hostgal]
    hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]

    if len(hosts_df) < 1:
        # if any features are nan, we can't use as input
        print(f"Some features are NaN for {IAU_name}. Skip!\n")
        return

    if show_host:
        print(
            f'http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color')

    hosts_df = hosts[feature_names_hostgal]
    hosts_df = pd.concat([hosts_df] * len(lc_properties_df), ignore_index=True)

    lc_and_hosts_df = pd.concat([lc_properties_df, hosts_df], axis=1)
    lc_and_hosts_df = lc_and_hosts_df.set_index('ztf_object_id')
    lc_and_hosts_df['raMean'] = hosts.raMean.values[0]
    lc_and_hosts_df['decMean'] = hosts.decMean.values[0]
    lc_and_hosts_df.to_csv(f'{df_path}/{lc_and_hosts_df.index[0]}_timeseries.csv')

    print(f"Saved results for {IAU_name}/{ztf_id_ref}!\n")


def panstarrs_image_filename(position, image_size=None, filter=None):
    """Query panstarrs service to get a list of image names
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :size : int: cutout image size in pixels.
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :filename: str: file name of the cutout
    """

    service = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
    url = (f'{service}?ra={position.ra.degree}&dec={position.dec.degree}'
           f'&size={image_size}&format=fits&filters={filter}')

    filename_table = pd.read_csv(url, delim_whitespace=True)['filename']
    return filename_table[0] if len(filename_table) > 0 else None


def panstarrs_cutout(position, filename, image_size=None, filter=None):
    """
    Download Panstarrs cutout from their own service
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    if filename:
        service = 'https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?'
        fits_url = (f'{service}ra={position.ra.degree}&dec={position.dec.degree}'
                    f'&size={image_size}&format=fits&red={filename}')
        fits_image = fits.open(fits_url)
    else:
        fits_image = None

    return fits_image


def host_pdfs(ztfid_ref, df, figure_path, ann_num, save_pdf=True, imsizepix=100, change_contrast=False):
    ref_name = df.ZTFID[0]
    data = df

    if save_pdf:
        pdf_path = f'{figure_path}/{ztfid_ref}_host_thumbnails_ann={ann_num}.pdf'
        pdf_pages = PdfPages(pdf_path)

    total_plots = len(df)
    rows = 3  # Number of rows in the subplot grid
    cols = 3  # Number of columns in the subplot grid
    num_subplots = rows * cols  # Total number of subplots in each figure
    num_pages = math.ceil(total_plots / num_subplots)

    for page in range(num_pages):
        fig, axs = plt.subplots(rows, cols, figsize=(6, 6))

        for i in range(num_subplots):
            index = page * num_subplots + i

            if index >= total_plots:
                break

            d = df.iloc[index]
            ax = axs[i // cols, i % cols]
            ax.set_xticks([])
            ax.set_yticks([])

            try:  # Has host assoc
                sc = SkyCoord(d['HOST_RA'], d['HOST_DEC'], unit=u.deg)

                outfilename = f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits"

                if os.path.isfile(outfilename):
                    print(f"Remove previously saved cutout {d['ZTFID']}_pscutout.fits to download a new one")
                    os.remove(outfilename)

                if not os.path.exists(outfilename):
                    filename = panstarrs_image_filename(sc, image_size=imsizepix, filter='r')
                    fits_image = panstarrs_cutout(sc, filename, image_size=imsizepix, filter='r')
                    fits_image.writeto(outfilename)

                wcs = WCS(f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits")

                imdata = fits.getdata(f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits")

                if change_contrast:
                    transform = AsinhStretch() + PercentileInterval(93)
                else:
                    transform = AsinhStretch() + PercentileInterval(99.5)

                bfim = transform(imdata)
                ax.imshow(bfim, cmap="gray", origin="lower")
                ax.set_title(f"{d['ZTFID']}", pad=0.1, fontsize=18)

            except:
                # Use a red square image when there is no data
                imdata = Image.new('RGB', (100, 100), color=(255, 0, 0))  # red
                ax.imshow(imdata, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f" ", pad=0.1, fontsize=18)

        # Remove axes labels
        for ax in axs.flat:
            ax.label_outer()
            ax.set_xticks([])
            ax.set_yticks([])

        # Reduce padding between subplots for a tighter layout
        plt.tight_layout(pad=0.1)

        plt.ion()
        plt.show()

        if save_pdf:
            pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()

        plt.close(fig)

    if save_pdf:
        pdf_pages.close()
        print(f"PDF saved at: {pdf_path}")

    plt.show()



############

def LAISS(l_or_ztfid_ref, lc_and_host_features, n=8, use_lc_for_ann_only_bool=False, use_ysepz_phot_snana_file=False,
          show_lightcurves_grid=False, show_hosts_grid=False, run_AD_model=False, savetables=False, savefigs=False):
    print("Running LAISS...")
    start_time = time.time()
    ann_num = n

    if use_ysepz_phot_snana_file:
        IAU_name = input("Input the IAU (TNS) name here, like: 2023abc\t")
        print("IAU_name:", IAU_name)
        ysepz_snana_fp = f"./ysepz_snana_phot_files/{IAU_name}_data.snana.txt"
        print(f"Looking for file {ysepz_snana_fp}...")

        # Initialize variables to store the values
        ra = None
        dec = None

        # Open the file for reading
        with open(ysepz_snana_fp, 'r') as file:
            # Read lines one by one
            for line in file:
                # Check if the line starts with '#'
                if line.startswith('#'):
                    # Split the line into key and value
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # Check if the key is 'RA' or 'DEC'
                        if key == '# RA':
                            ra = value
                        elif key == '# DEC':
                            dec = value

        SN_df = pd.read_csv(ysepz_snana_fp, comment="#", delimiter="\s+")
        SN_df = SN_df[
            (SN_df.FLT == "r-ZTF") | (SN_df.FLT == "g-ZTF") | (SN_df.FLT == "g") | (SN_df.FLT == "r")].reset_index(
            drop=True)
        SN_df["FLT"] = SN_df["FLT"].map({"g-ZTF": "g", "g": "g", "r-ZTF": "R", "r": "R"})
        SN_df = SN_df.sort_values('MJD')
        SN_df = SN_df.dropna()
        SN_df = SN_df.drop_duplicates(keep='first')
        SN_df = SN_df.drop_duplicates(subset=['MJD'], keep='first')
        print("Using S/N cut of 3...")
        SN_df = SN_df[SN_df.FLUXCAL >= 3 * SN_df.FLUXCALERR]  # SNR >= 3

    figure_path = f"./LAISS_run/{l_or_ztfid_ref}/figures"
    if savefigs:
        if not os.path.exists(figure_path):
            print(f"Making figures directory {figure_path}")
            os.makedirs(figure_path)

    table_path = f"./LAISS_run/{l_or_ztfid_ref}/tables"
    if savetables:
        if not os.path.exists(table_path):
            print(f"Making tables directory {table_path}")
            os.makedirs(table_path)

    needs_reextraction_for_AD = False
    l_or_ztfid_ref_in_dataset_bank = False

    host_df_ztf_id_l, host_df_ra_l, host_df_dec_l = [], [], []

    if l_or_ztfid_ref.startswith("ANT"):
        # Get locus data using antares_client
        try:
            locus = antares_client.search.get_by_id(l_or_ztfid_ref)
        except:
            print(f"Can't get locus. Check that {l_or_ztfid_ref} is a legimiate loci! Exiting...")
            return
        ztfid_ref = locus.properties['ztf_object_id']
        needs_reextraction_for_AD = True

        if 'tns_public_objects' not in locus.catalogs:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99
        else:
            tns = locus.catalog_objects['tns_public_objects'][0]
            tns_name, tns_cls, tns_z = tns['name'], tns['type'], tns['redshift']
        if tns_cls == "":
            tns_cls, tns_ann_z = "---", -99

        # Extract the relevant features
        try:
            locus_feat_arr = [locus.properties[f] for f in lc_and_host_features]
            print(locus.properties['raMean'], locus.properties['decMean'])
            print(
                f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={locus.properties['raMean']}+{locus.properties['decMean']}&filter=color\n")
            host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(locus.properties['raMean']), host_df_dec_l.append(
                locus.properties['decMean'])

        except:

            print(f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before...")
            if os.path.exists(
                    f"/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries/{ztfid_ref}_timeseries.csv"):
                print(f'{ztfid_ref} is already made. Continue!\n')
            else:
                print("Re-extracting features")
                if use_ysepz_phot_snana_file:
                    print("Using YSE-PZ SNANA Photometry file...")
                    extract_lc_and_host_features_YSE_snana_format(IAU_name=IAU_name, ztf_id_ref=l_or_ztfid_ref,
                                                                  yse_lightcurve=SN_df,
                                                                  ra=ra, dec=dec, show_lc=False, show_host=True)
                else:
                    extract_lc_and_host_features(ztf_id_ref=ztfid_ref,
                                                 use_lc_for_ann_only_bool=use_lc_for_ann_only_bool, show_lc=False,
                                                 show_host=True)

            try:
                lc_and_hosts_df = pd.read_csv(
                    f'/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries/{ztfid_ref}_timeseries.csv')
            except:
                print(f"couldn't feature space as func of time for {ztfid_ref}. pass.")
                return

            lc_and_hosts_df = lc_and_hosts_df.dropna()
            print(
                f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df['raMean']}+{lc_and_hosts_df['decMean']}&filter=color\n")
            host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(locus.properties['raMean']), host_df_dec_l.append(
                locus.properties['decMean'])
            try:
                lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
            except:
                print(f"{ztfid_ref} has some NaN LC features. Skip!")
                return

            anom_obj_df = pd.DataFrame(lc_and_hosts_df_120d.iloc[-1]).T  # last row of df to test "full LC only"
            locus_feat_arr = anom_obj_df.values[0]


    elif l_or_ztfid_ref.startswith("ZTF"):
        # Assuming you have a list of feature values
        # n = n+1 # because object in dataset chooses itself as ANN=0
        ztfid_ref = l_or_ztfid_ref

        try:
            locus_feat_arr = dataset_bank_orig.loc[ztfid_ref].values
            needs_reextraction_for_AD = True
            l_or_ztfid_ref_in_dataset_bank = True
            print(f"{l_or_ztfid_ref} is in dataset_bank")
            n = n + 1

        except:
            print(f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before...")
            if os.path.exists(
                    f"/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries/{l_or_ztfid_ref}_timeseries.csv"):
                print(f'{l_or_ztfid_ref} is already made. Continue!\n')

            else:
                print("Re-extracting LC+HOST features")
                try:
                    if use_ysepz_phot_snana_file:
                        print("Using YSE-PZ SNANA Photometry file...")
                        extract_lc_and_host_features_YSE_snana_format(IAU_name=IAU_name, ztf_id_ref=l_or_ztfid_ref,
                                                                      yse_lightcurve=SN_df,
                                                                      ra=ra, dec=dec, show_lc=False, show_host=True)
                    else:
                        extract_lc_and_host_features(ztf_id_ref=ztfid_ref,
                                                     use_lc_for_ann_only_bool=use_lc_for_ann_only_bool, show_lc=False,
                                                     show_host=True)
                except:
                    print(f"Can't extract features for {ztfid_ref}. Double check this object. Exiting...")
                    return

            try:
                lc_and_hosts_df = pd.read_csv(
                    f'/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries/{l_or_ztfid_ref}_timeseries.csv')
            except:
                print(f"couldn't feature space as func of time for {l_or_ztfid_ref}. pass.")
                return

            if not use_lc_for_ann_only_bool:
                print(
                    f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n")
                host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(
                    lc_and_hosts_df.iloc[0]['raMean']), host_df_dec_l.append(lc_and_hosts_df.iloc[0]['decMean'])
                lc_and_hosts_df = lc_and_hosts_df.dropna()  # if this drops all rows, that means something is nan from a 0 or nan entry (check data file)
                # lc_and_hosts_df = lc_and_hosts_df[lc_and_hosts_df.mjd_cutoff <= 59000]
                # print("lc_and_hosts_df", lc_and_hosts_df)

                try:
                    lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
                except:
                    print(f"{ztfid_ref} has some NaN LC features. Skip!")
                    return

                anom_obj_df = pd.DataFrame(lc_and_hosts_df_120d.iloc[-1]).T  # last row of df to test "full LC only"
                locus_feat_arr = anom_obj_df.values[0]

            if use_lc_for_ann_only_bool:
                try:
                    lc_only_df = lc_and_hosts_df.copy()
                    lc_only_df = lc_only_df.dropna()
                    lc_only_df = lc_only_df[feature_names_r_g]
                    lc_and_hosts_df_120d = lc_only_df.copy()

                    anom_obj_df = pd.DataFrame(lc_and_hosts_df_120d.iloc[-1]).T  # last row of df to test "full LC only"
                    locus_feat_arr = anom_obj_df.values[0]
                except:
                    print(f"{ztfid_ref} doesn't have enough g or r obs. Skip!")
                    return

        locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztfid_ref)
        try:
            tns = locus.catalog_objects['tns_public_objects'][0]
            tns_name, tns_cls, tns_z = tns['name'], tns['type'], tns['redshift']
        except:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99
        if tns_cls == "":
            tns_cls, tns_ann_z = "---", -99



    else:
        raise ValueError("Input must be a string (l or ztfid_ref) or a list of feature values")

    if not use_lc_for_ann_only_bool:
        # 1. Scale locus_feat_arr using the same scaler (Standard Scaler)
        scaler = preprocessing.StandardScaler()
        trained_PCA_feat_arr = np.load(f"./dataset_bank_60pca_annoy_index_feat_arr.npy", allow_pickle=True)
        # trained_PCA_feat_arr_scaled = np.load(f"./dataset_bank_60pca_annoy_index_feat_arr_scaled.npy", allow_pickle=True)
        # trained_PCA_feat_arr_scaled_pca = np.load(f"./dataset_bank_60pca_annoy_index_feat_arr_scaled_pca.npy", allow_pickle=True)
        trained_PCA_feat_arr_scaled = scaler.fit_transform(
            trained_PCA_feat_arr)  # scaler needs to be fit first to the same data as trained

        locus_feat_arr_scaled = scaler.transform([locus_feat_arr])  # scaler transform new data

        # 2. Transform the scaled locus_feat_arr using the same PCA model (60 PCs, RS=42)
        n_components = 60
        random_seed = 42
        pca = PCA(n_components=n_components, random_state=random_seed)
        trained_PCA_feat_arr_scaled_pca = pca.fit_transform(
            trained_PCA_feat_arr_scaled)  # pca needs to be fit first to the same data as trained
        locus_feat_arr_pca = pca.transform(locus_feat_arr_scaled)  # pca transform  new data

        # Create or load the ANNOY index
        index_nm = "./dataset_bank_60pca_annoy_index"  # 5k, 1000 trees
        index_file = "./dataset_bank_60pca_annoy_index.ann"  # 5k, 1000 trees
        index_dim = 60  # Dimension of the PCA index

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY LC+HOST PCA=60 index")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric='manhattan')
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(locus_feat_arr_pca[0], n=n, include_distances=True)
        ann_alerce_links = [f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes]
        ann_end_time = time.time()

    else:

        if l_or_ztfid_ref_in_dataset_bank:
            locus_feat_arr = locus_feat_arr[0:62]

        # 1, 2, 3. Don't use PCA at all. Just use LC features only + ANNOY index to find nearest neighbors
        # Create or load the ANNOY index

        # TODO: make argument for which file
        # index_nm = "dataset_bank_LCfeats_only_annoy_index" #5k, 1000 trees
        # index_file = "./dataset_bank_LCfeats_only_annoy_index.ann" #5k, 1000 trees

        # index_nm = "./bigbank_90k_LCfeats_only_annoy_index_100trees" #90k, 100 trees
        # index_file = "./bigbank_90k_LCfeats_only_annoy_index_100trees.ann" #90k, 100 trees

        index_nm = "./bigbank_90k_LCfeats_only_annoy_index_1000trees"  # 90k, 1000 trees
        index_file = "./bigbank_90k_LCfeats_only_annoy_index_1000trees.ann"  # 90k, 1000 trees
        index_dim = 62  # Dimension of the index

        print("Loading previously saved ANNOY LC-only index")
        print(index_file)
        index = annoy.AnnoyIndex(index_dim, metric='manhattan')
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(locus_feat_arr, n=n, include_distances=True)
        ann_alerce_links = [f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes]
        ann_end_time = time.time()

    # 4. Get TNS, spec. class of ANNs
    tns_ann_names, tns_ann_classes, tns_ann_zs = [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)
        try:
            ann_tns = ann_locus.catalog_objects['tns_public_objects'][0]
            tns_ann_name, tns_ann_cls, tns_ann_z = ann_tns['name'], ann_tns['type'], ann_tns['redshift']
        except:
            tns_ann_name, tns_ann_cls, tns_ann_z = "No TNS", "---", -99
        if tns_ann_cls == "":
            tns_ann_cls, tns_ann_z = "---", -99
        tns_ann_names.append(tns_ann_name), tns_ann_classes.append(tns_ann_cls), tns_ann_zs.append(tns_ann_z)
        host_df_ztf_id_l.append(idx_arr[i])

    #     if ztfid_ref=='ZTF21acmnpqa':
    #         tns_cls='SN Ia-91bg-like'

    # Print the nearest neighbors
    print("\t\t\t\t\t   ZTFID IAU_NAME SPEC Z")
    print(f"REF. : https://alerce.online/object/{ztfid_ref} {tns_name} {tns_cls} {tns_z}")

    ann_num_l = []
    for i, (al, iau_name, spec_cls, z) in enumerate(zip(ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs)):
        if l_or_ztfid_ref.startswith("ZTF"):
            if i == 0: continue
            print(f"ANN={i}: {al} {iau_name} {spec_cls}, {z}")
            ann_num_l.append(i)
        else:
            print(f"ANN={i + 1}: {al} {iau_name} {spec_cls} {z}")
            ann_num_l.append(i + 1)

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time = {round(ann_elapsed_time, 3)} s")
    print(f"\ntotal elapsed_time = {round(elapsed_time, 3)} s\n")

    if savetables:
        print("Saving reference+ANN table...")
        if l_or_ztfid_ref_in_dataset_bank:
            ref_and_ann_df = pd.DataFrame(
                zip(host_df_ztf_id_l, list(range(0, n + 1)), tns_ann_names, tns_ann_classes, tns_ann_zs),
                columns=['ZTFID', 'ANN_NUM', 'IAU_NAME', 'SPEC_CLS', 'Z'])
        else:
            ref_and_ann_df = pd.DataFrame(
                zip(host_df_ztf_id_l, list(range(0, n + 1)), [tns_name] + tns_ann_names, [tns_cls] + tns_ann_classes,
                    [tns_z] + tns_ann_zs),
                columns=['ZTFID', 'ANN_NUM', 'IAU_NAME', 'SPEC_CLS', 'Z'])
        ref_and_ann_df.to_csv(f"{table_path}/{ztfid_ref}_ann={ann_num}.csv", index=False)
        print(f"CSV saved at: {table_path}/{ztfid_ref}_ann={ann_num}.csv")

    #############
    if show_lightcurves_grid:
        print("Making a plot of stacked lightcurves...")

        if tns_z is None:
            tns_z = "None"
        elif isinstance(tns_z, float):
            tns_z = round(tns_z, 3)
        else:
            tns_z = tns_z

        if use_ysepz_phot_snana_file:
            try:
                df_ref = SN_df
            except:
                print("No timeseries data...pass!")
                pass

            fig, ax = plt.subplots(figsize=(9.5, 6))

            df_ref_g = df_ref[(df_ref.FLT == 'g') & (~df_ref.MAG.isna())]
            df_ref_r = df_ref[(df_ref.FLT == 'R') & (~df_ref.MAG.isna())]

            mjd_idx_at_min_mag_r_ref = df_ref_r[['MAG']].reset_index().idxmin().MAG
            mjd_idx_at_min_mag_g_ref = df_ref_g[['MAG']].reset_index().idxmin().MAG

            ax.errorbar(x=df_ref_r.MJD - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref],
                        y=df_ref_r.MAG.min() - df_ref_r.MAG, yerr=df_ref_r.MAGERR, fmt='o', c='r',
                        label=f'REF: {ztfid_ref}, d=0\n{tns_name},\t{tns_cls},\tz={tns_z}')
            ax.errorbar(x=df_ref_g.MJD - df_ref_g.MJD.iloc[mjd_idx_at_min_mag_g_ref],
                        y=df_ref_g.MAG.min() - df_ref_g.MAG, yerr=df_ref_g.MAGERR, fmt='o', c='g')

        else:
            ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztfid_ref)
            try:
                df_ref = ref_info.timeseries.to_pandas()
            except:
                print("No timeseries data...pass!")
                pass

            fig, ax = plt.subplots(figsize=(10, 6)) # tweaking figsize for display. Originally 9.5, 6

            df_ref_g = df_ref[(df_ref.ant_passband == 'g') & (~df_ref.ant_mag.isna())]
            df_ref_r = df_ref[(df_ref.ant_passband == 'R') & (~df_ref.ant_mag.isna())]

            mjd_idx_at_min_mag_r_ref = df_ref_r[['ant_mag']].reset_index().idxmin().ant_mag
            mjd_idx_at_min_mag_g_ref = df_ref_g[['ant_mag']].reset_index().idxmin().ant_mag

            ax.errorbar(x=df_ref_r.ant_mjd - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref],
                        y=df_ref_r.ant_mag.min() - df_ref_r.ant_mag, yerr=df_ref_r.ant_magerr, fmt='o', c='r',
                        label=f'REF: {ztfid_ref}, d=0\n{tns_name},\t{tns_cls},\tz={tns_z}')
            ax.errorbar(x=df_ref_g.ant_mjd - df_ref_g.ant_mjd.iloc[mjd_idx_at_min_mag_g_ref],
                        y=df_ref_g.ant_mag.min() - df_ref_g.ant_mag, yerr=df_ref_g.ant_magerr, fmt='o', c='g')

        markers = ['s', '*', 'x', 'P', '^', 'v', 'D', '<', '>', '8', 'p', 'x']
        consts = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

        if l_or_ztfid_ref_in_dataset_bank:
            ann_locus_l = ann_locus_l[1:]
            host_df_ztf_id_l = host_df_ztf_id_l
            ann_dists = ann_dists[1:]
            tns_ann_names = tns_ann_names[1:]
            tns_ann_classes = tns_ann_classes[1:]
            tns_ann_zs = tns_ann_zs[1:]

        for num, (l_info, ztfname, dist, iau_name, spec_cls, z) in enumerate(
                zip(ann_locus_l, host_df_ztf_id_l[1:], ann_dists, tns_ann_names, tns_ann_classes, tns_ann_zs)):
            try:

                # if ztfname=='ZTF21abaiicy':
                #    iau_name='2021lyi'
                #    spec_cls='SN Ia'
                #    z=0.07632

                df_knn = l_info.timeseries.to_pandas()

                df_g = df_knn[(df_knn.ant_passband == 'g') & (~df_knn.ant_mag.isna())]
                df_r = df_knn[(df_knn.ant_passband == 'R') & (~df_knn.ant_mag.isna())]

                mjd_idx_at_min_mag_r = df_r[['ant_mag']].reset_index().idxmin().ant_mag
                mjd_idx_at_min_mag_g = df_g[['ant_mag']].reset_index().idxmin().ant_mag

                ax.errorbar(x=df_r.ant_mjd - df_r.ant_mjd.iloc[mjd_idx_at_min_mag_r],
                            y=df_r.ant_mag.min() - df_r.ant_mag, yerr=df_r.ant_magerr,
                            fmt=markers[num], c='darkred', alpha=0.25,
                            label=f'ANN={num + 1}: {ztfname}, d={int(dist)}\n{iau_name},\t{spec_cls},\tz={round(z, 3)}')
                ax.errorbar(x=df_g.ant_mjd - df_g.ant_mjd.iloc[mjd_idx_at_min_mag_g],
                            y=df_g.ant_mag.min() - df_g.ant_mag, yerr=df_g.ant_magerr,
                            fmt=markers[num], c='darkgreen', alpha=0.25)
                # ax.text(df_ref_r.ant_mjd.iloc[-1]-df_ref_r.ant_mjd.iloc[0]+15, df_r.ant_mag[-1]-df_r.ant_mag.min(), s=f'ANN={num+1}: {has_tns_knn}   {tns_cls_knn}')

                plt.ylabel('Apparent Mag. + Constant')
                # plt.xlabel('Days of event') # make iloc[0]
                plt.xlabel('Days since peak ($r$, $g$ indep.)')  # (need r, g to be same)

                if use_ysepz_phot_snana_file:
                    if df_ref_r.MJD.iloc[0] - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref] <= 10:
                        plt.xlim((df_ref_r.MJD.iloc[0] - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref]) - 20,
                                 df_ref_r.MJD.iloc[-1] - df_ref_r.MJD.iloc[0] + 15)
                    else:
                        plt.xlim(2 * (df_ref_r.MJD.iloc[0] - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref]),
                                 df_ref_r.MJD.iloc[-1] - df_ref_r.MJD.iloc[0] + 15)

                else:
                    if df_ref_r.ant_mjd.iloc[0] - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref] <= 10:
                        plt.xlim((df_ref_r.ant_mjd.iloc[0] - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]) - 20,
                                 df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15)
                    else:
                        plt.xlim(2 * (df_ref_r.ant_mjd.iloc[0] - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]),
                                 df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15)

                plt.legend(frameon=False,
                           loc='upper right',
                           bbox_to_anchor=(0.58, 0.65, 0.5, 0.5), #tweaking for display. originally (0.52, 0.85, 0.5, 0.5)
                           ncol=5, #tweaking for display. originally 3
                           columnspacing=0.75,
                           prop={'size': 8})

                plt.grid(True)
                # plt.ylim(-2.25,0.25)


            except Exception as e:
                print(f"Something went wrong with plotting {ztfname}! Error is {e}. Continue...")

        if savefigs:
            print("Saving stacked lightcurve...")
            plt.savefig(f'{figure_path}/{ztfid_ref}_stacked_lightcurve_ann={ann_num}.pdf', dpi=300, bbox_inches='tight')
            print(f"PDF saved at: {figure_path}/{ztfid_ref}_stacked_lightcurve_ann={ann_num}.pdf")

        plt.show()

    ##############
    if show_hosts_grid:
        print("\nGenerating hosts grid plot...")

        dataset_bank_orig_w_hosts_ra_dec = pd.read_csv(
            '../loci_dbs/alerce_cut/dataset_bank_orig_w_hosts_ra_dec_5472objs.csv.gz', compression='gzip', index_col=0)
        for j, ztfid in enumerate(host_df_ztf_id_l):  # first entry is reference, which we already calculated
            if j == 0:
                try:
                    print(
                        f"REF.  ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={host_df_ra_l[0]}+{host_df_dec_l[0]}&filter=color")
                    continue
                except:
                    print(
                        f"REF.  ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].raMean}+{dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].decMean}&filter=color")
                    pass
            h_ra, h_dec = dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].raMean, dataset_bank_orig_w_hosts_ra_dec.loc[
                ztfid].decMean
            host_df_ra_l.append(h_ra), host_df_dec_l.append(h_dec)
            if j == 0: continue
            print(f"ANN={j} ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={h_ra}+{h_dec}&filter=color")
        host_5ann_df = pd.DataFrame(zip(host_df_ztf_id_l, host_df_ra_l, host_df_dec_l),
                                    columns=['ZTFID', 'HOST_RA', 'HOST_DEC'])
        if savefigs:
            print("Saving host thumbnails pdf...")
            host_pdfs(ztfid_ref=ztfid_ref, df=host_5ann_df, figure_path=figure_path, ann_num=ann_num, save_pdf=True)
        else:
            host_pdfs(ztfid_ref=ztfid_ref, df=host_5ann_df, figure_path=figure_path, ann_num=ann_num, save_pdf=False)

        if savetables:
            print("Saving host thumbnails table...")
            host_5ann_df.to_csv(f"{table_path}/{ztfid_ref}_host_thumbnails_ann={ann_num}.csv", index=False)
            print(f"CSV saved at: {table_path}/{ztfid_ref}_host_thumbnails_ann={ann_num}.csv")

    ########################
    if run_AD_model:
        print("\nRunning AD Model!...")
        if needs_reextraction_for_AD:
            print("Needs re-extraction for full timeseries.")
            print("Checking if made before...")
            if os.path.exists(
                    f"/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries/{ztfid_ref}_timeseries.csv"):
                print(f'{ztfid_ref} is already made. Continue!\n')
            else:
                print("Re-extracting LC+HOST features")
                if use_ysepz_phot_snana_file:
                    print("Using YSE-PZ SNANA Photometry file...")
                    extract_lc_and_host_features_YSE_snana_format(IAU_name=IAU_name, ztf_id_ref=l_or_ztfid_ref,
                                                                  yse_lightcurve=SN_df,
                                                                  ra=ra, dec=dec, show_lc=False, show_host=True)
                else:
                    extract_lc_and_host_features(ztf_id_ref=ztfid_ref,
                                                 use_lc_for_ann_only_bool=use_lc_for_ann_only_bool, show_lc=False,
                                                 show_host=True)

            try:
                lc_and_hosts_df = pd.read_csv(
                    f'/Users/patrickaleo/Desktop/Illinois/LAISS-antares/repo/tables/custom/timeseries/{ztfid_ref}_timeseries.csv')
            except:
                print(f"couldn't feature space as func of time for {ztfid_ref}. pass.")
                return

            try:
                print(
                    f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n")
            except:
                pass

            lc_and_hosts_df = lc_and_hosts_df.dropna()

            try:
                lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
            except:
                print(f"{ztfid_ref} has some NaN LC features. Skip!")

        if use_ysepz_phot_snana_file:
            plot_RFC_prob_vs_lc_yse_IAUid(clf=clf,
                                          IAU_name=IAU_name,
                                          anom_ztfid=l_or_ztfid_ref,
                                          anom_spec_cls=tns_cls,
                                          anom_spec_z=tns_z,
                                          anom_thresh=50,
                                          lc_and_hosts_df=lc_and_hosts_df,
                                          lc_and_hosts_df_120d=lc_and_hosts_df_120d,
                                          yse_lightcurve=SN_df,
                                          savefig=savefigs,
                                          figure_path=figure_path)
        else:
            plot_RFC_prob_vs_lc_ztfid(clf=clf,
                                      anom_ztfid=ztfid_ref,
                                      anom_spec_cls=tns_cls,
                                      anom_spec_z=tns_z,
                                      anom_thresh=50,
                                      lc_and_hosts_df=lc_and_hosts_df,
                                      lc_and_hosts_df_120d=lc_and_hosts_df_120d,
                                      ref_info=locus,
                                      savefig=savefigs,
                                      figure_path=figure_path)

# Example

# LAISS(l_or_ztfid_ref="ZTF18abydmfv",
#       lc_and_host_features=lc_and_host_features,
#       n=8,
#       use_lc_for_ann_only_bool=True, # currently doesn't work with YSE_snana_format or ANT IDs
#       use_ysepz_phot_snana_file=False,
#       show_lightcurves_grid=False,
#       show_hosts_grid=False,
#       run_AD_model=False,
#       savetables=False,
#       savefigs=False)

LAISS(l_or_ztfid_ref=args.l_or_ztfid_ref,
      lc_and_host_features=lc_and_host_features,
      n=args.num_ann,
      use_lc_for_ann_only_bool=args.use_lc_for_ann_only_bool, # currently doesn't work with YSE_snana_format or ANT IDs
      use_ysepz_phot_snana_file=args.use_ysepz_phot_snana_file,
      show_lightcurves_grid=args.show_lightcurves_grid,
      show_hosts_grid=args.show_hosts_grid,
      run_AD_model=args.run_AD_model,
      savetables=args.savetables,
      savefigs=args.savefigs)