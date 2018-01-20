#!/bin/bash

for ((i=2010;i<=2014;i++)) do
wget "https://wwwdasis.samhsa.gov/dasis2/teds_pubs/TEDS/Admissions/${i}/TEDSA_${i}_PUF_CSV.zip"
unzip TEDSA_${i}_PUF_CSV.zip
rm TEDSA_${i}_PUF_CSV.zip
done

#for ((i=2010;i<=2014;i++)) do
#wget "https://wwwdasis.samhsa.gov/dasis2/teds_pubs/TEDS/Discharges/TED_D_${i}/teds_d_${i}_csv.zip"

#for i in 2010 2012 2014
#do
#wget "https://wwwdasis.samhsa.gov/dasis2/nmhss/mh${i}_puf_csv.zip"
#unzip mh${i}_puf_csv.zip
#rm mh${i}_puf_csv.zip
#done


echo "download done"
