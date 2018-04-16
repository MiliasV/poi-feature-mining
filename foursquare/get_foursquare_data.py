import foursquare as fs


def setup():
    CLIENT_ID = "OLQQ0OBYQMIDKLLQE5CYMZSYNDVEUJOITEOEMLZ5GJ5AZN2F"
    CLIENT_SECRET = "PEX54BQ2TD00AHSFJCDR5VP11YEUNJOFSKRXZYFGHGRRN00L"
    client = fs.Foursquare(client_id=CLIENT_ID,
                                   client_secret=CLIENT_SECRET)
    return client


def get_venues_by_ll(client, ll, radius):
    venues = client.venues.search({'intent': 'browse', 'limit': 10000, 'radius':radius,
                                  'll': ll})
    return venues


def get_addr_from_venue(venue):
    if "location" in venue and "address" in venue["location"]:
        addr = venue["location"]["address"]
        num = [x for x in addr.split() if any(char.isdigit() for char in x)]
        street = ' '.join([x for x in addr.split() if not any(char.isdigit() for char in x)])
        if num:
            num = num[0]
        return street, num
    else:
        return [], []


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


def create_dict_by_category_id(venues_categories):
    venues_category_dict = {}
    print(venues_categories)
    for venue_category in venues_categories[0]["categories"]:
        venues_category_dict[venue_category["id"]] = venue_category
        # for id_special_category in venue_category:
        #     # print(id_special_category)
        #     # print(venue_category[id_special_category])
        #     venues_category_dict[venue_category[id_special_category]] = venue_category
        #     print("ok")
    for i in venues_category_dict:
        print(i, venues_category_dict[i])
    return venues_category_dict


if __name__ == '__main__':
    #Initialization
    # # client = setup()
    venues_categories = client.venues.categories()# ["categories"]
    for i in venues_categories:
        print(venues_categories[i])
        for j in venues_categories[i]:
            print(j)
            print(len(venues_categories))
    categories_dict = create_dict_by_category_id(venues_categories)

    # Difference example
    # !Geo-location varies between sources!
    # google_maps = 37.985976,         23.732598
    # foursquare  = 37.98600059717795, 23.73266215976166
    # osm         = 37.9859432,        23.7325542
    # GET Venues
    # Choose Long, Lat
    coords = {'ll': '37.9859432,23.7325542'}
    # Choose radius
    radius = 10
    # Get the venues in this radius
    venues = get_venues_by_ll(ll, radius) # ll: '32.324,25.2424'

    for venue in venues["venues"]:
        if "categories" in venue and venue["categories"]:
            print(venue)
            id_category = venue["categories"][0]["id"]
            print(venue["name"], "--category-->", venue["categories"][0]["name"])

    photos = client.venues.photos(VENUE_ID="4efe05970e01089c53e3764a", params={})

