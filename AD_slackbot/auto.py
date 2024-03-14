from source import *
from astropy.time import Time
import tempfile
import os
import glob
from build_rec import post as pst
from build_rec import build_rec as bs
import time
import argparse

import antares_client
import datetime
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Run LAISS AD. Ex: python3 auto.py 2 D05R7RK4K8T')
# Add command-line arguments for input, new data, and output file paths
parser.add_argument('lookback_t', help='lookback_t')
parser.add_argument('anom_thresh', help='anom_thresh')
parser.add_argument('channel', help='slack channel ID')
args = parser.parse_args()

# Get current date
current_date = datetime.datetime.now()
year = current_date.year
month = current_date.month
day = current_date.day

# Calculate today's MJD
today_mjd = calculate_mjd(year, month, day)
print("Today's Modified Julian Date:", today_mjd)

lookback_t = float(args.lookback_t)
channel = str(args.channel)
print(f"Looking back to all objects tagged within MJD {today_mjd}-{today_mjd - lookback_t}:")

anom_thresh = float(args.anom_thresh) # anomaly threshold
print(f"Using Anomaly Threshold of: {anom_thresh}%")

# Get list of tagged loci
LAISS_RFC_AD_loci = antares_client.search.search(
    {
        "query": {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "tags": [
                                "LAISS_RFC_AD_filter"
                            ]
                        }
                    }
                ],
                "must": {
                    "range": {
                        "properties.newest_alert_observation_time": {
                            "gte": today_mjd - lookback_t
                        }
                    }
                }
            }
        }
    }
)

# https://stackoverflow.com/questions/1528237/how-to-handle-exceptions-in-a-list-comprehensions
def catch(func, *args, handle=lambda e : e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)

LAISS_RFC_AD_locus_ids = [catch(lambda : l.locus_id) for l in LAISS_RFC_AD_loci] # to catch requests.exceptions.JSONDecodeError: [Errno Expecting ',' delimiter]
print(f"Considering {len(LAISS_RFC_AD_locus_ids)} candidates...")

g50_antid_l, g50_tns_name_l, g50_tns_cls_l, g50_anom_score_l, g50_ra_l, g50_dec_l = [], [], [], [], [], []
for l in LAISS_RFC_AD_locus_ids:
    if l.startswith("ANT2023"):  # only take objects from this year
        locus = antares_client.search.get_by_id(l)
        if 'tns_public_objects' not in locus.catalogs:
            if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= anom_thresh:
                g50_antid_l.append(l), g50_tns_name_l.append("No TNS"), g50_tns_cls_l.append("---"), g50_anom_score_l.append(locus.properties['LAISS_RFC_anomaly_score'])
                g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)

        else:
            try:
                tns = locus.catalog_objects['tns_public_objects'][0]
                tns_name, tns_cls = tns['name'], tns['type']
            except:
                print(f"{l} likely is on TNS but is outside of 1arcsec matching for catalogs...Check! Anom score: {locus.properties['LAISS_RFC_anomaly_score']}")

            if tns_cls == '': tns_cls = "---"
            if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= anom_thresh:
                #print(f"https://antares.noirlab.edu/loci/{l}", tns_name, tns_cls, locus.properties['LAISS_RFC_anomaly_score'])
                g50_antid_l.append(l), g50_tns_name_l.append(tns_name), g50_tns_cls_l.append(tns_cls), g50_anom_score_l.append(locus.properties['LAISS_RFC_anomaly_score'])
                g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)

# only print objects with 'no' or "NA" for both Stern and Jarrett AGN thresholds
print("Applying further cuts to remove possible AGN...")
final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l = wise_diag(antid=g50_antid_l, tns_name=g50_tns_name_l,
                                                                            tns_cls=g50_tns_cls_l, anom_score=g50_anom_score_l, ra=g50_ra_l,
                                                                            dec=g50_dec_l)


def run(post=True):
    ps= []
    if post:
        ps.append(f"ZTF Anomalies (re-)tagged within MJD {today_mjd - lookback_t} to {today_mjd}:\n")
        for antaresID, tns_name, tns_cls, anom_score in zip(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l):
            try:
                ant = bs(antaresID=antaresID, tns_name=tns_name, tns_cls=tns_cls, anom_score=anom_score)
            except:
                print(f"Some error while trying to post for: {antaresID} {tns_name} {tns_cls} {anom_score}. Skip!")
                continue
            ps.append(ant.string)
        ps = '\n'.join(ps)
        pst(ps,channel=channel) # C03STCB0ACA = anomaly-detection channel; 'D05R7RK4K8T' == Bot specific channel LAISS_AD_bot for testing


    return 0

def save_objects(file):
    if not os.path.exists(file):
        df = pd.DataFrame(zip(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l),
                          columns=['ANTID', 'TNS_Name', 'Spec_Class'])
        df.to_csv(file)
    else:
        df = pd.read_csv(file)
        df2 = pd.DataFrame(zip(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l),
                          columns=['ANTID', 'TNS_Name', 'Spec_Class'])
        df.set_index('ANTID', inplace=True)
        df2.set_index('ANTID', inplace=True)
        # Concatenate the two dataframes along the rows
        merged_df = pd.concat([df, df2])
        # keep the last occurrence of each duplicate row -- (most up to date)
        merged_df = merged_df.drop_duplicates(keep='last')
        merged_df.to_csv(file) # overwrite the old file with unique new + old objects



# def run_sched(sep= 86400): # 24 hours in seconds
#     while True:
#         t1 = time.monotonic()
#         run()
#         t2 = time.monotonic()
#         td = t2-t1
#         time.sleep(sep-td) # calculation offset so it runs at the same time every day
#     return 0


if __name__ == '__main__':
    # #print('running')
    # parser = argparse.ArgumentParser(description='Run LAISS AD')
    # # Add command-line arguments for input, new data, and output file paths
    # parser.add_argument('lookback_t', help='lookback_t')
    # args = parser.parse_args()
    #
    # lookback_t = args.lookback_t

    run()
    save_objects(file='./anomalies_db.csv')
    #run_sched()

def main():
    #print('running')
    run()
    save_objects(file='./anomalies_db.csv')
    #run_sched()
    return 0