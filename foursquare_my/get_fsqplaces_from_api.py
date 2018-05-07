import foursquare as fs
import sys
sys.path.append("..")
import foursquare_my.fsq_config as fsq_config
import foursquare_my.get_fsq_data as get_fsq_data
import get_map_points_to_search
import pprint
import pois_storing_functions
import simplejson as json


def setup():
    CLIENT_ID = fsq_config.cl_id
    CLIENT_SECRET = fsq_config.cl_sec
    client = fs.Foursquare(client_id=CLIENT_ID,
                           client_secret=CLIENT_SECRET)
    return client


def get_value_if_exist(dict, value):
    if value in dict:
        return dict[value]
    else:
        return None


def extract_data_from_json(fsq_json):
    res = {}
    print(fsq_json["location"])
    res["fsq_id"] = fsq_json["id"]
    res["name"] = fsq_json["name"]
    res["pop_times"] = get_value_if_exist(fsq_json, "popular")
    res["website"] = get_value_if_exist(fsq_json, "url")
    res["street"], res["street_num"] = get_fsq_data.get_addr_from_venue(fsq_json)
    if "stats" in fsq_json:
        res["check_ins"] = get_value_if_exist(fsq_json["stats"], "checkinsCount")
        res["users_count"] = get_value_if_exist(fsq_json["stats"], "usersCount")
        res["tip_count"] = get_value_if_exist(fsq_json["stats"], "tipCount")
        res["visits_count"] = get_value_if_exist(fsq_json["stats"], "visitsCount")
    res["rating"] = get_value_if_exist(fsq_json, "rating")
    res["price"] = get_value_if_exist(fsq_json, "price")
    if "likes" in fsq_json:
        res["likes_count"] = get_value_if_exist(fsq_json["likes"], "count")
    if "photos" in fsq_json:
        res["photos_count"] = get_value_if_exist(fsq_json["photos"], "count")
    if "contact" in fsq_json:
        res["facebook"] = get_value_if_exist(fsq_json["contact"], "facebookName")
        res["twitter"] = get_value_if_exist(fsq_json["contact"], "twitter")
        res["phone"] = get_value_if_exist(fsq_json["contact"], "formattedPhone")

    return res


def insert_data(session, GTable, search_lat, search_lng, google_json, ogc_fid,
                count_places, count_duplicates):
    fsq_id, name, types, reviews_count, photos_count, pop_times, lat, lng, website, street_lng, \
    street_srt, street_num_lng, street_num_srt = extract_data_from_json(fsq_json)
    json_string = json.dumps(fsq_json)
    geo = 'POINT({} {})'.format(lng, lat)



if __name__ == '__main__':
    cl = setup()
    c, rad, logfile = get_map_points_to_search.config_parameters_for_searching("fsq")
    last_searched_id = pois_storing_functions.get_last_id_from_logfile(logfile)

    # setup table and session
    # define which table (fsq table, count table)
    # session, FTable, CTable = pois_storing_functions.setup_db("fsq_ams_center_40",
    #                                                           "fsq_ams_center_40_count")
    pp = pprint.PrettyPrinter(indent=4)
    # get hierarchy categories tree
    categories_tree = cl.venues.categories()
    print(get_fsq_data.find_key_path_from_value(categories_tree["categories"], "Cantonese Restaurant"))

    #pp.pprint((categories_tree))

    # For each point --> search nearby in google
    for ogc_fid, point_lat, point_lng in c:
        count_places = 0
        count_duplicates = 0
        print("POINT: ", ogc_fid, point_lat, point_lng)
        # keep the id of the last searched point
        get_map_points_to_search.log_last_searched_point(logfile, ogc_fid)
        ll = str(point_lat) + ", " + str(point_lng)
        # define which table
        session, FTable, CTable = pois_storing_functions.setup_db("fsq_ams_center_40",
                                                                  "fsq_ams_center_40_count")
        # specify which categories to search for!
        # food = 4d4b7105d754a06374d81259
        # arts and entertainment = 4d4b7104d754a06370d81259
        # college & uni = 4d4b7105d754a06372d81259
        # nightlife spot = 4d4b7105d754a06376d81259
        # outdoor & recreation = 4d4b7105d754a06377d81259
        # food and drink shop = 4bf58dd8d48988d1f9941735
        categories = "4d4b7105d754a06374d81259,4d4b7104d754a06370d81259,4d4b7105d754a06372d81259," \
                     "4d4b7105d754a06376d81259,4bf58dd8d48988d1f9941735"
        venues = get_fsq_data.get_venues_by_ll(cl, ll, rad, categories)
        for venue in venues["venues"]:
            fsq_json = get_fsq_data.get_venue_details(cl, venue["id"])
            #count_places, count_duplicates =\
            insert_data(session, FTable, point_lat, point_lng, fsq_json, ogc_fid,
                            count_places, count_duplicates)
            #insert_count_data(session, CTable, ogc_fid, count_places, count_duplicates)