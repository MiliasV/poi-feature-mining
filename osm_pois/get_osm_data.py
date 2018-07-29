# Functions to get OSM POIs' data from sqlite db


# OSM categories
# pois_ids = get_osm_data.get_ids_by_category(c, "'amenity'", "('cinema', 'nightclub', 'theatre', "
#                                                           "'bar', 'cafe', 'fast_food', 'pub', 'restaurant'"
#                                                           "'pharmacy', 'hospital', 'university','school'"
#                                                           "'cemetery', 'police', 'church', 'archaeologogical_site')")

def get_osm_name_addr_info(c, poi_id):
    # choose what to use for matching!
    name = get_name_from_id(c, poi_id)
    type = get_type_from_id(c, poi_id)
    info = get_info_by_id(c, poi_id)
    addr_street = get_street_from_id(c, poi_id)
    addr_street_num = get_street_num_from_id(c, poi_id)
    if addr_street and addr_street_num:
        addr_street = addr_street + " " + addr_street_num
    return name, type, addr_street, info


def get_info_by_id(c, search_id):
    c.execute("SELECT * FROM nodes_tags WHERE id={object_id}".format(object_id=search_id))
    nodes_tags = c.fetchall()
    if nodes_tags:
        c.execute("SELECT * FROM nodes WHERE id={object_id}".format(object_id=search_id))
        nodes = c.fetchall()
        return nodes, nodes_tags
    else:
        return [], []


def get_all_pois(c):
    return c.execute("SELECT * FROM nodes").fetchall()


def get_all_ids(c):
    return c.execute("SELECT id FROM nodes").fetchall()


def get_ids_by_category(c, category, t):
    return c.execute("SELECT id FROM nodes_tags WHERE k LIKE {cat} AND v IN {type}".format(cat=category, type=t)).fetchall()


def get_name_from_id(c, id):
    names = c.execute("SELECT v FROM nodes_tags WHERE id={object_id} AND k LIKE 'name%'".format(object_id=id)).fetchone()
    return names[0]


def get_addr_from_id(c, id):
    return c.execute("SELECT k,v FROM nodes_tags WHERE id={object_id} AND k LIKE 'addr%'".format(object_id=id)).fetchall()


def get_street_from_id(c, id):
    street = c.execute("SELECT v FROM nodes_tags WHERE id={object_id} AND k LIKE 'addr:street%'".format(object_id=id)).fetchall()
    if street:
        return street[0][0]
    return []


def get_street_num_from_id(c, id):
    street_num = c.execute("SELECT v FROM nodes_tags WHERE id={object_id} AND k LIKE 'addr:housenumber%'".format(object_id=id)).fetchall()
    if street_num:
        return street_num[0][0]
    return []


def get_type_from_id(c, id):
    return c.execute("SELECT v FROM nodes_tags WHERE id={object_id} AND k LIKE 'amenity%'".format(object_id=id)).fetchall()


def get_website_from_id(c, id):
    web_list = c.execute("SELECT v FROM nodes_tags WHERE id={object_id} AND k LIKE 'website%'".format(object_id=id)).fetchall()
    if web_list:
        return web_list[0]
    else:
        return web_list


def get_lat_long_from_id(c, id):
    return c.execute("SELECT * FROM nodes WHERE id={object_id}".format(object_id=id)).fetchall()


def get_data_for_matching(osm_info, c, poi_id):
    osm_info["id"] = poi_id
    osm_info["street"] = get_street_from_id(c, poi_id)
    osm_info["street_num"] = get_street_num_from_id(c, poi_id)
    osm_info["type"] = get_type_from_id(c, poi_id)
    osm_info["website"] = get_website_from_id(c, poi_id)
    # get longtitude and latitude
    osm_info["ll"] = get_lat_long_from_id(c, poi_id)
    return osm_info
