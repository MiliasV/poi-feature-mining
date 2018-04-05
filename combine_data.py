import sqlite3
import osm_pois.sqlite_processing as osm_sqlite
import foursquare_folder.get_foursquare_data as fsq
from difflib import SequenceMatcher
from nltk.corpus import wordnet


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_osm_name_addr_info(c, poi_id):
    # choose what to use for matching!
    name = osm_sqlite.get_name_from_id(c, poi_id)
    type = osm_sqlite.get_type_from_id(c, poi_id)
    info = osm_sqlite.get_info_by_id(c, poi_id)
    addr_street = osm_sqlite.get_street_from_id(c, poi_id)
    addr_street_num = osm_sqlite.get_street_num_from_id(c, poi_id)
    if addr_street and addr_street_num:
        addr_street = addr_street + " " + addr_street_num
    return name, type, addr_street, info


def get_similarity_score(osm_name, fs_name, osm_addr, fs_addr):
    num_attr = 0
    score = 0
    if osm_name and fs_name:
        score += similar(osm_name, fs_name)
        num_attr += 1
    if osm_addr and fs_addr:
        score += similar(osm_addr, fs_addr)
        num_attr += 1
    return float(score)/num_attr


def match_osm_foursquare(c, poi_id, ll, radius):
    # get info by osm
    osm_name, osm_type, osm_addr, osm_info = get_osm_name_addr_info(c, poi_id)
    print("# OSM: ", osm_name, osm_type, osm_addr, osm_info)

    # get info by foursquare
    fsq_client = fsq.setup()
    # first filter: in that radius
    fsq_venues = fsq.get_venues_by_ll(fsq_client, ll, radius)

    if not fsq_venues["venues"]:
        #print("No venues found")
        return 0
    for venue in fsq_venues["venues"]:
        # get info from foursquare
        fs_name = venue["name"]
        fs_addr = fsq.get_addr_from_venue(venue)

        # check name similarity
        name_similarity = similar(osm_name, fs_name)
        addr_similarity = similar(osm_addr, fs_addr)
        similarity = get_similarity_score(osm_name, fs_name, osm_addr, fs_addr)
        if similarity > -0.001:
            print("Similarity score =  ", similarity *100, "%")
            print("Names:", osm_name, fs_name, name_similarity)
            print("Addr:", osm_addr, fs_addr, addr_similarity)
        else:
            print("Nothing similar found")
        #photos = client.venues.photos(VENUE_ID="4efe05970e01089c53e3764a", params={})

        return 0

# databases
ams_db ='../../databases/amsterdam_map.osm.db'
ath_db = '../../databases/athens.osm.db'

# select osm db
DB = ams_db

# connect to osm db
conn = sqlite3.connect(DB)
c = conn.cursor()

# select pois'types
pois_ids = osm_sqlite.get_ids_by_category(c, "'amenity'", "('cinema', 'nightclub', 'theatre', "
                                                          "'bar', 'cafe', 'fast_food', 'pub', 'restaurant'"
                                                          "'pharmacy', 'hospital', 'university','school'"
                                                          "'cemetery', 'police', 'church', 'archaeologogical_site')")
count = 0
# choose radius
rad = 10
# For each poi
for poi in pois_ids[10:20]:
    print("##########  POI NUMBER: ", count, "####################################################################")
    count+=1
    # continue if there is at least a name
    osm_name = osm_sqlite.get_name_from_id(c, poi[0])
    if osm_name:
        lat_long = osm_sqlite.get_lat_long_from_id(c, poi[0])
        ll = str(lat_long[0][1]) + "," + str(lat_long[0][2])
        match_osm_foursquare(c, poi[0], ll, radius=rad)# > 0.5:
        #     print("MATCHED !")
        print("########################################################################################")

# Foursquare Enrichment

#osm_name[0][2], osm_type, osm_addr,