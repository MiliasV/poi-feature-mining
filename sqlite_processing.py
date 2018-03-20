import sqlite3
conn = sqlite3.connect('/home/bill/athens.osm.db')

c = conn.cursor()

k_type = "amenity"
v_type = "addr"

c.execute("SELECT * FROM nodes")#  WHERE v NOT LIKE '%s'" % ( v_type))
nodes = c.fetchall()

c.execute("SELECT * FROM nodes_tags")#  WHERE v NOT LIKE '%s'" % ( v_type))
nodes_tags = c.fetchall()

poi_list = []
poi_dict = {}
for poi in nodes_tags:
    if poi[0] not in poi_dict:
        poi_dict[poi[0]] = []
    poi_dict[poi[0]].append([poi[1], poi[2]])

for node in nodes:
    if node[0] in poi_dict:
        poi_dict[node[0]].append([node[1], node[2]])

flat_poi_list = []

for poi_id in poi_dict:
    flat_poi = [item for sublist in poi_dict[poi_id] for item in sublist]
    if "amenity" not in flat_poi:
        print(poi_dict[poi_id])

#for poi_id in poi_dict:
#    print(poi_dict[poi_id])
