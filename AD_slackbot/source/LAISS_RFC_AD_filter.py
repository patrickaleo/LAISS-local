# Get list of new anomaly candidates (>=50% anomaly score) and post as Slack Bot
# written by Patrick Aleo

import antares_client
import datetime
import os
import sys
import time



def calculate_mjd(year, month, day):
    # Calculate Julian Date
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    julian_date = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    
    # Calculate Modified Julian Date
    modified_jd = julian_date - 2400000.5
    return modified_jd

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
          "properties.newest_alert_observation_time" : {
            "gte": today_mjd - 1
          }
        }
       }
      }
    }
  }
)
LAISS_RFC_AD_locus_ids = [l.locus_id for l in LAISS_RFC_AD_loci]

g50_antid_l, g50_tns_name_l, g50_tns_cls_l, g50_ra_l, g50_dec_l = [], [], [], [], []
for l in LAISS_RFC_AD_locus_ids:
    locus = antares_client.search.get_by_id(l)
    if 'tns_public_objects' not in locus.catalogs: 
        if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= 50:
            print(f"https://antares.noirlab.edu/loci/{l}", "No TNS", "---", locus.properties['LAISS_RFC_anomaly_score'])
            g50_antid_l.append(l), g50_tns_name_l.append("No TNS"), g50_tns_cls_l.append("---") 
            g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)

    else: 
        tns = locus.catalog_objects['tns_public_objects'][0]
        tns_name, tns_cls = tns['name'], tns['type']
        if tns_cls is '': tns_cls = "---"
        if 'LAISS_RFC_anomaly_score' in locus.properties and locus.properties['LAISS_RFC_anomaly_score'] >= 50:
            print(f"https://antares.noirlab.edu/loci/{l}", tns_name, tns_cls, locus.properties['LAISS_RFC_anomaly_score'])
            g50_antid_l.append(l), g50_tns_name_l.append(tns_name), g50_tns_cls_l.append(tns_cls) 
            g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)
    