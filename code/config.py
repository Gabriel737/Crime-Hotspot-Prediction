## Vancouver crime data config file

# Self-defined crime categories
CRIME_CATS = ['Break and Enter', 'Homicide', 'Mischief', 'Assualt', 'Theft', 'Vehicle Collision']

# Crime types present in Vancouver crime data
CRIME_TYPES = ['Break and Enter Commercial','Break and Enter Residential/Other',
               'Homicide','Mischief','Offence Against a Person','Other Theft',
               'Theft from Vehicle','Theft of Bicycle','Theft of Vehicle',
               'Vehicle Collision or Pedestrian Struck (with Fatality)',
               'Vehicle Collision or Pedestrian Struck (with Injury)']

# Crime type to crime category mapping
TYPE2CAT = {'Break and Enter Commercial':'Break and Enter',
            'Break and Enter Residential/Other':'Break and Enter',
            'Homicide':'Homicide', 'Mischief':'Mischief',
            'Offence Against a Person':'Assualt',
            'Theft from Vehicle':'Theft',
            'Theft of Bicycle':'Theft',
            'Theft of Vehicle':'Theft',
            'Vehicle Collision or Pedestrian Struck (with Fatality)':'Vehicle Collision',
            'Vehicle Collision or Pedestrian Struck (with Injury)':'Vehicle Collision'}

# Coordinates for bounding box
BB_P1 = {'lat': 49.30112, 'long': -123.02245}
BB_P2 = {'lat': 49.18444, 'long': -123.02245}
BB_P3 = {'lat': 49.18444, 'long': -123.20071}
BB_P4 = {'lat': 49.30112, 'long': -123.20071}

# Bounding box edge length (in kms)
BB_DIST = 13

# Bounding box cell length (in kms)
BB_CELL_LEN = 0.5




