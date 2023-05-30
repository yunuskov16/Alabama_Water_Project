# Alabama_Water_Project
All the files associated with creating a webtool to assess the cost of decentralized water treatment options

Notes about the data:
1. The building shapefile data comes from a github repository that is maintained by Microsoft as part of their US Building Footprints database. Everytime the code is run it will draw from the most up to date building data on their github page
2. The shapefiles for us aquifers come from the USGS groundwater data downloaded in 2021
3. The elevation data comes from the py3dep package, which is directly connected to the USGS 3D elevation program
4. The cost numbers are all specific to Uniontown, Alabama in 2023. The paper "Optimizing Scale for Decentralized Wastewater Treatment: A Tool to Address Failing Wastewater Infrastructure in the United States" gives the exact methodology as to how we came up with the cost numbers (eventually we will need to link the cost numbers to a database which is regularly maintained and updated).
