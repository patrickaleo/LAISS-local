from source import *
from astropy.time import Time
import tempfile
import os
import glob
from build_rec import post as pst
from build_rec import build_rec as bs
import time

import antares_client
import datetime

# Get current date
current_date = datetime.datetime.now()
year = current_date.year
month = current_date.month
day = current_date.day

# Calculate today's MJD
today_mjd = calculate_mjd(year, month, day)
print("Today's Modified Julian Date:", today_mjd)

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
                            "gte": today_mjd - 1
                        }
                    }
                }
            }
        }
    }
)
LAISS_RFC_AD_locus_ids = [l.locus_id for l in LAISS_RFC_AD_loci]
print(f"Considering {len(LAISS_RFC_AD_locus_ids)} candidates...")

g50_antid_l, g50_tns_name_l, g50_tns_cls_l, g50_anom_score_l, g50_ra_l, g50_dec_l = [], [], [], [], [], []
for l in LAISS_RFC_AD_locus_ids:
    if l.startswith("ANT2023"):  # only take objects from this year
        locus = antares_client.search.get_by_id(l)
        if 'tns_public_objects' not in locus.catalogs:
            if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= 50:
                #print(f"https://antares.noirlab.edu/loci/{l}", "No TNS", "---", locus.properties['LAISS_RFC_anomaly_score'])
                g50_antid_l.append(l), g50_tns_name_l.append("No TNS"), g50_tns_cls_l.append("---"), g50_anom_score_l.append(locus.properties['LAISS_RFC_anomaly_score'])
                g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)

        else:
            tns = locus.catalog_objects['tns_public_objects'][0]
            tns_name, tns_cls = tns['name'], tns['type']
            if tns_cls == '': tns_cls = "---"
            if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= 50:
                #print(f"https://antares.noirlab.edu/loci/{l}", tns_name, tns_cls, locus.properties['LAISS_RFC_anomaly_score'])
                g50_antid_l.append(l), g50_tns_name_l.append(tns_name), g50_tns_cls_l.append(tns_cls), g50_anom_score_l.append(locus.properties['LAISS_RFC_anomaly_score'])
                g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)

# only print objects with 'no' or "NA" for both Stern and Jarrett AGN thresholds
final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l = wise_diag(antid=g50_antid_l, tns_name=g50_tns_name_l,
                                                                            tns_cls=g50_tns_cls_l, anom_score=g50_anom_score_l, ra=g50_ra_l,
                                                                            dec=g50_dec_l)


def run(post=True):
    ps= []
    if post:
        for antaresID, tns_name, tns_cls, anom_score in zip(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l):
            ant = bs(antaresID=antaresID, tns_name=tns_name, tns_cls=tns_cls, anom_score=anom_score)
            ps.append(ant.string)
        ps = '\n'.join(ps)
        pst(ps,channel='D05R7RK4K8T') # LAISS_AD_bot channel for testing

    return 0


# def run_sched(sep= 86400): # 24 hours in seconds
#     while True:
#         t1 = time.monotonic()
#         run()
#         t2 = time.monotonic()
#         td = t2-t1
#         time.sleep(sep-td) # calculation offset so it runs at the same time every day
#     return 0


if __name__ == '__main__':
    print('running')
    run()
    #run_sched()

def main():
    print('running')
    run()
    #run_sched()
    return 0