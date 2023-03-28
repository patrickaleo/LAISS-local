#!/bin/bash

for i in {0..89}; do
    declination=$(printf "%d" $i)
    filename="hlsp_ps1-psc_ps1_gpc1_${declination}_multi_v1_cat.fits"
    url="https://archive.stsci.edu/hlsps/ps1-psc/${filename}"
    wget -v -nH -np "$url"
done

for i in {-31..-1}; do
    declination=$(printf "%d" $i)
    filename="hlsp_ps1-psc_ps1_gpc1_${declination}_multi_v1_cat.fits"
    url="https://archive.stsci.edu/hlsps/ps1-psc/${filename}"
    wget -v -nH -np "$url"
done
