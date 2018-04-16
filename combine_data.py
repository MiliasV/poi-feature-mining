import sqlite3
import osm_pois.get_osm_data as get_osm_data
import foursquare_folder.get_foursquare_data as fsq
from difflib import SequenceMatcher
from nltk.corpus import wordnet


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_similarity_score(osm_info, source_info):
    num_attr = 0
    score = 0
    for attr in ["name", "street", "street_num", "website", "type"]:
        if osm_info.get(attr) and source_info.get(attr):
            # check both long and short type description from foursquare
            if attr == "type":
                score += max((similar(str(osm_info[attr]), str(source_info[attr][0])),
                               similar(str(osm_info[attr]), str(source_info[attr][1]))
                              ))
            else:
                score += similar(str(osm_info[attr]), str(source_info[attr]))
            num_attr += 1
    return float(score)/num_attr, num_attr


def get_best_match_from_foursquare(c, poi_id, ll, radius, osm_info):
    # translate ll for fourquare API
    ll = str(ll[0][1]) + "," + str(ll[0][2])
    # define fs client
    fsq_client = fsq.setup()
    # first filter: in that radius
    fsq_venues = fsq.get_venues_by_ll(fsq_client, ll, radius)

    if not fsq_venues["venues"]:
        print("No venues found in radius = ", radius, " m")
        return 0

    for venue in fsq_venues["venues"]:
        fs_info = {}
        # get info from foursquare
        fs_info["name"] = venue["name"]
        fs_info["street"], fs_info["street_num"] = fsq.get_addr_from_venue(venue)
        fs_info["type"] = fsq.get_type_from_venue(venue)
        fs_info["website"] = fsq.get_website_from_venue(venue)
        print(venue)
        similarity, attr_num = get_similarity_score(osm_info, fs_info)
        if similarity > 0.1:
            print("Similarity score =  ", similarity *100, "%")
            print("Attributes counted:", attr_num)
            print("# OSM: ", osm_info)
            print("# FS INFO: ", fs_info)

        else:
            print("Nothing similar found")
        #photos = client.venues.photos(VENUE_ID="4efe05970e01089c53e3764a", params={})


if __name__ == "__main__":
    # databases
    ams_db ='../../databases/ams_bb.db'
    ath_db = '../../databases/ath_bb.db'
    # select osm db
    DB = ams_db
    # connect to osm db
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # select which pois'types we investigate
    pois_ids = get_osm_data.get_ids_by_category(c, "'amenity'", "('cinema', 'nightclub', 'theatre', "
                                                              "'bar', 'cafe', 'fast_food', 'pub', 'restaurant'"
                                                              "'pharmacy', 'hospital', 'university','school'"
                                                              "'cemetery', 'police', 'church', 'archaeologogical_site')")
    # choose radius (how far from the osm ll)
    rad = 20
    count = 0
    # For each poi
    for poi in pois_ids[10:20]:
        osm_info = {}
        poi_id = poi[0]
        print("##########  POI ( ", poi_id," ) | Loop: ", count, "####################################################################")
        count += 1
        # get osm info
        osm_info["name"] = get_osm_data.get_name_from_id(c, poi_id)
        # continue if there is at least a name (restriction)
        if osm_info["name"]:
            osm_info["street"] = get_osm_data.get_street_from_id(c, poi_id)
            osm_info["street_num"] = get_osm_data.get_street_num_from_id(c, poi_id)
            osm_info["type"] = get_osm_data.get_type_from_id(c, poi_id)
            osm_info["website"] = get_osm_data.get_website_from_id(c, poi_id)
            # get longtitude and latitude
            osm_info["ll"] = get_osm_data.get_lat_long_from_id(c, poi_id)
            get_best_match_from_foursquare(c, poi[0], osm_info["ll"], rad, osm_info)# > 0.5:
            #     print("MATCHED !")
            print("########################################################################################")

    # Foursquare Enrichment

    #osm_name[0][2], osm_type, osm_addr,