import numpy as np
import json
import copy
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astroquery.vizier import Vizier

def check_altaz(alt,az,star_pair_altaz):
    found = False
    for i in range(len(star_pair_altaz)):
        if ((abs(star_pair_altaz[i][0]-alt) < 16) and (abs(star_pair_altaz[i][1]-az)) < 31):
            found = True
            break
    return found
        

def compute_pair_list(nb_pairs, Vmag_max, t0, sep_range, elevation_min):
    rubin_site = EarthLocation.of_site('Rubin Observatory')
    
    Vizier.ROW_LIMIT=-1
    catalogs = Vizier.get_catalogs('V/50') #this the Yale Bright Star Catalog
    table = catalogs[0] #get the table

    selection_vmag = (table['Vmag'] < Vmag_max)
    ra = table[selection_vmag]['RAJ2000']
    dec = table[selection_vmag]['DEJ2000']
    #stars_radec = SkyCoord(ra,dec,unit=(u.hourangle,u.deg),frame='fk5') #YBSC is in FK5 ref frame
    stars_radec = SkyCoord(ra,dec,unit=(u.hourangle,u.deg),frame='fk5') #YBSC is in FK5 ref frame
    #print(stars_radec)

    time = Time(t0) #in UTC
    stars_altaz = stars_radec.transform_to(AltAz(obstime=time,location=rubin_site))
    selection_alt = (stars_altaz.alt > elevation_min*u.deg)

    star_pair_radec = []
    star_pair_altaz = []
    pair_count = 0
    pair_found = False

    for i,refstar in enumerate(stars_altaz[selection_alt]):
        sep = refstar.separation(stars_altaz[selection_alt])
        selection_sep = (sep > sep_range[0]*u.deg ) & (sep < sep_range[1]*u.deg)
        if len(sep[selection_sep]) > 0:
            for j in range(len(sep[selection_sep])):
                ra1 = refstar.transform_to('icrs').ra.deg
                dec1 = refstar.transform_to('icrs').dec.deg
                ra2 = stars_altaz[selection_alt][selection_sep][j].transform_to('icrs').ra.deg
                dec2 = stars_altaz[selection_alt][selection_sep][j].transform_to('icrs').dec.deg
                alt = (refstar.alt.deg + stars_altaz[selection_alt][selection_sep][j].alt.deg)/2. 
                az = (refstar.az.deg + stars_altaz[selection_alt][selection_sep][j].az.deg)/2. 
                if check_altaz(alt,az,star_pair_altaz):
                    continue
                else:
                    star_pair_radec.append([ra1,dec1,ra2,dec2])
                    star_pair_altaz.append([alt,az])
                    pair_count = pair_count + 1
                    pair_found = True
                if pair_count >= nb_pairs:
                    break 
            else: #this is a little trick to break out of both loops
                continue
            break     
  
    if pair_found == False:
        print("No pairs were found")
    return star_pair_radec,star_pair_altaz

def update_block(file_name_out, pair_list):
    with open("BLOCK-250.json", "r") as block_file_ref:
        data = json.load(block_file_ref)

    last_command = data["scripts"].pop() #remove last command, will be added later 
    dummy = data["scripts"].pop()
    dummy = data["scripts"].pop() # now we have removed the last three elements
    #now we substitute the first pair into the block
    data["scripts"][0]["parameters"]["target_name"] = ""
    data["scripts"][0]["parameters"]["slew_icrs"] = {"ra":str(pair_list[0][0]),"dec":str(pair_list[0][1])}
    data["scripts"][4]["parameters"]["target_name"] = ""
    data["scripts"][4]["parameters"]["slew_icrs"] = {"ra":str(pair_list[0][2]),"dec":str(pair_list[0][3])}
    #we now proceed to create 4 new entries per pair (2 track targets and 2 sleeps)
    for i in range(len(pair_list)-1):
        for j in range(2): 
            data["scripts"].append(copy.deepcopy(data["scripts"][4]))
            data["scripts"].append(copy.deepcopy(data["scripts"][5]))
        data["scripts"][(i*4)+6]["parameters"]["target_name"] = ""
        data["scripts"][(i*4)+6]["parameters"]["slew_icrs"] = {"ra":str(pair_list[i+1][0]),"dec":str(pair_list[i+1][1])}
        data["scripts"][(i*4)+8]["parameters"]["target_name"] = ""
        data["scripts"][(i*4)+8]["parameters"]["slew_icrs"] = {"ra":str(pair_list[i+1][2]),"dec":str(pair_list[i+1][3])}
    data["scripts"].append(last_command)

    #for i in range(len(pair_list)-1):

    with open(file_name_out, "w") as block_file_out:
        json.dump(data, block_file_out)

def main():
    # Define some configuration variables
    Vmag_max = 5 # maximum magnitude of the stars in the list
    t0 = '2024-3-8 01:00:00' #in UTC, ideally the central time at which observations will take place
    sep_range = [3.3,3.7] #in degrees, minimum separation between stars
    elevation_min = 30. #  minimum elevation for stars in the pair, NOTE that some might go below
                    # this elevation threshold as the sky rotates during the BLOCK
    max_nb_pairs = 50
    (pair_list,altaz) = compute_pair_list(max_nb_pairs, Vmag_max, t0, sep_range, elevation_min)
    if len(pair_list) < 1:
        return -1
    update_block("BLOCK-250_updated.json",pair_list)
    for i in range(len(pair_list)):
        print(i,pair_list[i],altaz[i])

if __name__ == "__main__":
    main()

