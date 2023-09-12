# Get list of new anomaly candidates (>=50% anomaly score) and post as Slack Bot
# written by Patrick Aleo

import antares_client
import datetime
import os
import sys
import time
from collections import defaultdict

import numpy as np
from astropy.io import ascii,fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astroquery.ipac.irsa import Irsa


def calculate_mjd(year, month, day):
    # Calculate Julian Date
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    julian_date = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    
    # Calculate Modified Julian Date
    modified_jd = julian_date - 2400000.5
    return modified_jd


# input either an RA and Dec or a list of RAs and list of Declinations
def wise_diag(antid, tns_name, tns_cls, anom_score, ra, dec):
	
	# jarrett box
	xarr = np.arange(2.2, 4.25, 0.05)
	lower_line = (0.1 * xarr) + 0.38
	
	if isinstance(ra, float) == True:
		ra = np.array([ra])
		dec = np.array([dec])	
	else:
		ra = np.array(ra)
		dec = np.array(dec)
		
	
	coord = SkyCoord(ra, dec, unit=u.deg)		
	
	w1 = np.zeros(len(ra))
	w2 = np.zeros(len(ra))
	w3 = np.zeros(len(ra))
	
	stern = np.array(['NA'] * len(ra), dtype=object)
	jarrett = np.array(['NA'] * len(ra), dtype=object)		
			
	i=0
	while i < len(ra): 
		table = Irsa.query_region(coord[i], catalog='allwise_p3as_psd', radius=0.000833333*u.deg) # 3 arcsec
		if len(table) > 0:
		
			w1[i] = table[0]['w1mpro'] #using the closest match found in the table
			w2[i] = table[0]['w2mpro']
			w3[i] = table[0]['w3mpro']
			
			w12_obj = w1[i] - w2[i]
			w23_obj = w2[i] - w3[i]
			
			if w12_obj>0.8:
				stern[i] = 'yes'
			else:
				stern[i] = 'no'	
			if ((w23_obj > 2.2) & (w23_obj < 4.2) & (w12_obj > (0.1*(w23_obj)+0.38)) & (w12_obj < 1.7)):
				jarrett[i] = 'yes'	
			else:
				jarrett[i] = 'no'		
		
			i=i+1
		
		else:
			#print('No match in WISE for', coord[i], 'within 3 arcsec')
			i=i+1		
		
	w1_w2 = w1-w2
	w2_w3 = w2-w3
		
	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	#
	# ax.axvline(x=2.2, ymin=0.24, ymax=0.68, c='k', label = 'Jarrett et al. 2011')
	# ax.axvline(x=4.2, ymin=0.32, ymax=0.68, c='k')
	# ax.axhline(1.7, xmin=0.3, xmax=0.8, color='k')
	# ax.scatter(w2_w3, w1_w2, marker='o', s=80, facecolor='#61a5e3', edgecolor='k', lw=1., zorder=12)
	# for i, txt in enumerate(tns_name):
	# 	ax.annotate(txt, (w2_w3[i]-0.1, w1_w2[i]+0.05))
	# ax.plot(xarr, lower_line, c='k')
	# ax.axhline(0.8, c='gray', ls='--', label='Stern et al. 2012')
	# ax.legend(loc=2, fontsize=12)
	# ax.set_xlim(1,5)
	# ax.set_ylim(0,2.5)
	# ax.set_xlabel('W2-W3', fontsize=14)
	# ax.set_ylabel('W1-W2', fontsize=14)
	# plt.show()
	
	
	# print('RA', 'Dec', 'Stern', 'Jarrett')
	i=0
	final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l = [], [], [], []
	print("Final anomaly candidates are...")
	while i < len(coord):
		if (stern[i] == "no" or stern[i] == "NA") and (jarrett[i] == "no" or jarrett[i] == "NA"):
			print(f"https://antares.noirlab.edu/loci/{antid[i]}", tns_name[i], tns_cls[i], anom_score[i], stern[i], jarrett[i])
			final_cand_antid_l.append(antid[i]), final_cand_tns_name_l.append(tns_name[i]), final_cand_tns_cls_l.append(tns_cls[i]), final_cand_anom_score_l.append(anom_score[i])
			i = i + 1
		else:
			continue

	return final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l
	

# # Get current date
# current_date = datetime.datetime.now()
# year = current_date.year
# month = current_date.month
# day = current_date.day
#
# # Calculate today's MJD
# today_mjd = calculate_mjd(year, month, day)
# print("Today's Modified Julian Date:", today_mjd)
#
# # Get list of tagged loci
# LAISS_RFC_AD_loci = antares_client.search.search(
#     {
#   "query": {
#     "bool": {
#       "filter": [
#         {
#           "terms": {
#             "tags": [
#               "LAISS_RFC_AD_filter"
#             ]
#           }
#         }
#       ],
#         "must": {
#         "range": {
#           "properties.newest_alert_observation_time" : {
#             "gte": today_mjd - 1
#           }
#         }
#        }
#       }
#     }
#   }
# )
# LAISS_RFC_AD_locus_ids = [l.locus_id for l in LAISS_RFC_AD_loci]
# print(f"Considering {len(LAISS_RFC_AD_locus_ids)} candidates...")
#
# g50_antid_l, g50_tns_name_l, g50_tns_cls_l, g50_ra_l, g50_dec_l = [], [], [], [], []
# for l in LAISS_RFC_AD_locus_ids:
#     if l.startswith("ANT2023"): # only take objects from this year
#         locus = antares_client.search.get_by_id(l)
#         if 'tns_public_objects' not in locus.catalogs:
#             if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= 50:
#                 print(f"https://antares.noirlab.edu/loci/{l}", "No TNS", "---", locus.properties['LAISS_RFC_anomaly_score'])
#                 g50_antid_l.append(l), g50_tns_name_l.append("No TNS"), g50_tns_cls_l.append("---")
#                 g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)
#
#         else:
#             tns = locus.catalog_objects['tns_public_objects'][0]
#             tns_name, tns_cls = tns['name'], tns['type']
#             if tns_cls == '': tns_cls = "---"
#             if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= 50:
#                 print(f"https://antares.noirlab.edu/loci/{l}", tns_name, tns_cls, locus.properties['LAISS_RFC_anomaly_score'])
#                 g50_antid_l.append(l), g50_tns_name_l.append(tns_name), g50_tns_cls_l.append(tns_cls)
#                 g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)
#
# # only print objects with 'no' or "NA" for both Stern and Jarrett AGN thresholds
# print("g50_tns_name_l", g50_tns_name_l)
# final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l = wise_diag(antid=g50_antid_l, tns_name=g50_tns_name_l, tns_cls=g50_tns_cls_l, ra=g50_ra_l, dec=g50_dec_l)
# #print(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l)
