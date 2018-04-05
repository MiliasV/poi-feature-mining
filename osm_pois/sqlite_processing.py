import sqlite3


def get_info_by_id(c, search_id):
    c.execute("SELECT * FROM nodes_tags WHERE id={object_id}".format(object_id=search_id))
    nodes_tags = c.fetchall()
    if nodes_tags:
        c.execute("SELECT * FROM nodes WHERE id={object_id}".format(object_id=search_id))
        nodes = c.fetchall()
        return nodes, nodes_tags
    else:
        return [], []


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


def get_lat_long_from_id(c, id):
    return c.execute("SELECT * FROM nodes WHERE id={object_id}".format(object_id=id)).fetchall()



# poi_ids = get_ids_by_category(c, "'amenity'")
#
# for id in poi_ids:
#     nodes, nodes_tags = get_info_by_id(c, id[0])
#     if nodes:
#         print(nodes_tags)
#         print(nodes)

# print(nodes)
# c = conn.cursor()
#_type = "amenity"
#v_type = "addr"
# = conn.cursor()
# = 1682242134
#c.execute("SELECT * FROM nodes WHERE id={search_id}".format(search_id=n))

#odes = c.fetchall()
#or i in nodes:
#   print(i)
#rint(nodes)
#print(nodes)
#c.execute("SELECT * FROM nodes_tags")#  WHERE v NOT LIKE '%s'" % ( v_type))
#nodes_tags = c.fetchall()


# Creation ofpoi dictionaries - needed?
# poi_list = []
# poi_dict = {}
# for poi in nodes_tags:
#     if poi[0] not in poi_dict:
#         poi_dict[poi[0]] = []
#     poi_dict[poi[0]].append([poi[1], poi[2]])
#
# for node in nodes:
#     if node[0] in poi_dict:
#         poi_dict[node[0]].append([node[1], node[2]])
#
# flat_poi_list = []
#
# for poi_id in poi_dict:
#     flat_poi = [item for sublist in poi_dict[poi_id] for item in sublist]
#     if "amenity" in flat_poi:
#         print(poi_dict[poi_id])

