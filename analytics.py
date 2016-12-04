import csv
import json

yli_geo_train_path = ('/home/sneha/Documents/git/geolocation_attributes/raw/mediaeval2016_placing_train_photo')

node_count = {}
i=0
with open(yli_geo_train_path) as tsv:
    reader = csv.reader(tsv, delimiter='\t')
    for row in reader:
        i = i+1
        print i
        places = row[4].split(',')
        print places
#         for place in places:
#             node = place.split(':')[2]
#             if node not in node_count.keys():
#                 node_count[node] = 1
#             else:
#                 node_count[node] += 1
#
# with open('nodes.json', 'w') as fp:
#     json.dump(node_count, fp)
