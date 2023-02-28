import foursquare as fs


def get_venues_by_ll(client, ll, radius, categories):
    venues = client.venues.search({'intent': 'browse', 'limit': 50, 'radius':radius,
                                  'll': ll, 'categoryId': categories})
    return venues


def get_venue_details(client, venue_id):
    details = client.venues(venue_id)
    return details["venue"]


# logic --> https://stackoverflow.com/questions/15210148/get-parents-keys-from-nested-dictionary?
#           utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def find_key_path_from_value(d, value):
    for category_dict in d:
        if category_dict["categories"]:
            p = find_key_path_from_value(category_dict["categories"], value)
            if p:
                return p + [category_dict["name"]]
        if category_dict["name"] == value:
            return [value]


def get_dict_with_scored_venues(fsq_venues, get_similarity_score, source_info):
    fs_info = {}
    if not fsq_venues["venues"]:
        #print("No venues found in radius = ", radius, " m")
        return 0
    for venue in fsq_venues["venues"]:
        id = venue["id"]
        fs_info[id] = get_data_for_matching(venue)
        fs_info[id]["score"], fs_info[id]["num_attr"], fs_info[id]["sim_dict"] =\
            get_similarity_score(source_info, fs_info[id])
    return fs_info


def get_addr_from_venue(venue):
    if "location" in venue and "address" in venue["location"]:
        addr = venue["location"]["address"]
        num = [x for x in addr.split() if any(char.isdigit() for char in x)]
        street = ' '.join([x for x in addr.split() if not any(char.isdigit() for char in x)])
        if num:
            num = num[0]
        return street, num
    else:
        return None, None


def get_type_from_venue(venue):
    if "categories" in venue:
        for cat_dict in venue["categories"]:
            # if primary category
            if cat_dict["primary"]:
                return cat_dict['name'], cat_dict["shortName"]
        return []


def get_website_from_venue(venue):
    if "url" in venue:
        return venue["url"]
    else:
        return []


def get_data_for_matching(venue):
    match_dict = {}
    # get info from foursquare
    #fs_info[venue["id"]] = {}
    match_dict["lat"] = venue["location"]["lat"]
    match_dict["lng"] = venue["location"]["lng"]
    match_dict["name"] = venue["name"]
    match_dict["street"], match_dict["street_num"] = get_addr_from_venue(venue)
    match_dict["type"] = get_type_from_venue(venue)
    match_dict["website"] = get_website_from_venue(venue)
    return match_dict


if __name__ == '__main__':
    #Initialization
    # # client = setup()
    print("ok")
    # Difference example
    # !Geo-location varies between sources!
    # google_maps = 37.985976,         23.732598
    # foursquare  = 37.98600059717795, 23.73266215976166
    # osm         = 37.9859432,        23.7325542
    # GET Venues
    # Choose Long, Lat


