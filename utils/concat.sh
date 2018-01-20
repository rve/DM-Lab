head -1 TEDSA_2010_PUF.csv > final.csv
for filename in $(ls TEDS*.csv); do sed 1d $filename >> final.csv; done

echo 'concat done'
