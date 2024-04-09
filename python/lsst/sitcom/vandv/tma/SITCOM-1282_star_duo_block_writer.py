# # Create a block with star pairs for star tracker tests
#
# This script will create a block, based on BLOCK-250, that would create a series
# of tracking and slewing events between pairs of stars, plus turning on and off the 
# star tracker CCDs
#
# Author: Nacho Sevilla
# https://rubinobs.atlassian.net/browse/SITCOM-1282
#

import numpy as np
import json
import copy
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
from optparse import OptionParser
import warnings
warnings.filterwarnings("ignore")

def plot_altaz(star_pair_altaz):
    """ Function to create a plot of the altitude-elevation distribution of the star pairs
    Parameters
    ----------
    star_pair_altaz: 2D list with floats (degrees)
        contains a nested list with alt-az positions, each row with one altitude, azimuth  
        corresponding to the approximate position of the pair
    Returns
    -------
    Saves a plot.
    """
    
    alts = [star[0] for star in star_pair_altaz]
    azs = [star[1] for star in star_pair_altaz]
    azs_rad = np.radians(np.array(azs))

    fig=plt.figure(figsize=(8,8))
    ax1 = plt.subplot(121, projection='polar')
    ax1.set_title("Star pair distribution")
    ax1.invert_yaxis() # This puts 90 degrees in the center, 0 at the outside.
    ax1.scatter(azs_rad, alts, color='red', marker='x', label="Star pair position")
    #ax1.legend(loc='lower right')    
    plt.savefig("Stair_pair_distribution.png",bbox_inches='tight')

def check_altaz(ref_alt,ref_az,star_pair_altaz,alt_interval,az_interval):
    """ Function to check whether the star pair average altitude azimuth 
    has already been included in a previous pair of the same execution
    Parameters
    ----------
    ref_alt: float (degrees)
        Pair altitude
    ref_az: float (degrees)
        Pair azimuth
    star_pair_altaz: 2D list with floats (degrees)
        contains a nested list with alt-az positions, each row with one altitude, azimuth  
        corresponding to the approximate position of the pair
    alt_interval: altitude interval (degrees)
        minimum altitude interval between pairs to be considered for the BLOCK
    az_interval: azimuth interval (degrees)
        minimum azimuth interval between pairs to be considered for the BLOCK
    Returns
    -------
    found: bool
        Boolean indicating True if the alt-az combination is within any of the intervals, meaning that the
        combination has been explored already
    """ 
    found = False
    for i in range(len(star_pair_altaz)):
        if ((abs(star_pair_altaz[i][0]-ref_alt) < alt_interval) 
            and (abs(star_pair_altaz[i][1]-ref_az)) < az_interval):
            found = True
            break
    return found
        

def compute_pair_list(nb_pairs, Vmag_max, t0, sep_range, elevation_min):
    """ Function to compute pairs of stars

    Parameters
    ----------
    nb_pairs: int
        Maximum number of pair of stars
    Vmag_max: float
        V maximum magnitude limit to retrieve from reference star catalog (Yale 5th Bright Star Catalog)
    t0: string 
        contains the UTC time at which the pair positions will be calculated (format '2024-3-8 01:00:00')
        This precludes from choosing very long blocks so that these positions do not change too much.
    sep_range: list with 2 floats (degrees)
        minimum and maximum allowed separation between the stars
    elevation_min: float (degrees)
        minimum elevation for the stars to be included in the list for possible pairings
    
    Returns
    -------
    star_pair_radec: 2D list with floats (degrees)
        contains a nested list with right ascension, declination of the pairs of stars
        in the format [ra1,dec1,ra2,dec2] for each row
    star_pair_altaz: 2D list with floats (degrees)
        contains a nested list with alt-az positions, each row with one altitude, azimuth  
        corresponding to the approximate position of the pair    
    """
    rubin_site = EarthLocation.of_site('Rubin Observatory')
    
    Vizier.ROW_LIMIT=-1
    catalogs = Vizier.get_catalogs('V/50') #this the Yale Bright Star Catalog
    table = catalogs[0] #get the table

    #select all stars brighter than Vmag_max
    selection_vmag = (table['Vmag'] < Vmag_max)
    ra = table[selection_vmag]['RAJ2000']
    dec = table[selection_vmag]['DEJ2000']
    stars_radec = SkyCoord(ra,dec,unit=(u.hourangle,u.deg),frame='fk5') #YBSC is in FK5 ref frame

    time = Time(t0) #in UTC
    stars_altaz = stars_radec.transform_to(AltAz(obstime=time,location=rubin_site))
    #additionally select all stars with an altitude above elevation_min
    selection_alt = (stars_altaz.alt > elevation_min*u.deg)

    star_pair_radec = []
    star_pair_altaz = []
    pair_count = 0
    pair_found = False
    alt_interval = 15 # degrees, minimum interval between pairs in altitude 
    az_interval = 30 # degrees, minimum interval between pairs in azimuth

    # for every selected star, pair it with those that are within sep_range 
    # this will create duplicate pairs which will be selected out with the 
    # check_altaz function outside this function
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
                if check_altaz(alt,az,star_pair_altaz,alt_interval,az_interval):
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

    plot_altaz(star_pair_altaz)
    if pair_found == False:
        print("No pairs were found")
    return star_pair_radec,star_pair_altaz

def update_block(file_name_out, pair_list):
    """ Function to rewrite BLOCK-250 with a new one with the computed star pairs
    Parameters
    ----------
    file_name_out: string
        BLOCK file name
    pair_list: 2D list with floats (degrees)
        contains a nested list with right ascension, declination of the pairs of stars
        in the format [ra1,dec1,ra2,dec2] for each row

    Returns
    -------
    Creates a new BLOCK.
    """
    # this a bit ad-hoc function, to be substituted by 
    # functions in ts-observing 
    
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

    with open(file_name_out, "w") as block_file_out:
        json.dump(data, block_file_out)

def main():

    """Run code with options"""
    
    usage = "%prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--Vmag",
        dest="Vmag_max",
        help="maximum magnitude of the stars in the list",
        default=5,
        type="float",
    )
    parser.add_option(
        "--t0",
        dest="t0",
        help="in UTC, central time at which observations will take place",
        default= '2024-3-8 01:00:00',
        type="string",
    )
    parser.add_option(
        "--minsep",
        dest="min_sep",
        help="minimum separation between stars",
        default=3.3,
        type="float",
    )
    parser.add_option(
        "--maxsep",
        dest="max_sep",
        help="maximum separation between stars",
        default=3.7,
        type="float",
    )
    parser.add_option(
        "--minalt",
        dest="elevation_min",
        help="minimum elevation for stars in the pair, NOTE that some might go below this elevation threshold as the sky rotates during the BLOCK",
        default=30.,
        type="float",
    )
    parser.add_option(
        "--maxpairs",
        dest="max_nb_pairs",
        help="maximum number of pairs to consider",
        default=50,
        type="int",
    )
    
    (options, args) = parser.parse_args()

    (pair_list,altaz) = compute_pair_list(options.max_nb_pairs, 
                                          options.Vmag_max, 
                                          options.t0, 
                                          [options.min_sep, options.max_sep], 
                                          options.elevation_min)
    
    if len(pair_list) < 1:
        return -1
    update_block("BLOCK-250_updated.json",pair_list)
#    for i in range(len(pair_list)):
#        print(i,pair_list[i],altaz[i])

if __name__ == "__main__":
    main()

