# script that cross references images from YLI_GEO to YFCC100M dataset and writes the corresponding metadata to a txt file
# utility function uncomment when required
# def binary_search(sample, key):
#     l = 0
#     r = len(sample) - 1
#     m = (r-l)/2
#     import pdb; pdb.set_trace();
#     while(l<r):
#         if(sample[m]==key):
#             return True
#         elif(key < sample[m]):
#             r = m
#             m = (r-l)/2
#         elif(key > sample[m]):
#             l = m
#             m=(r-l)/2
#     return False

import csv
import os
import sys
import urllib
from PIL import Image

csv.field_size_limit(sys.maxsize)

yfcc100m_path = ('/home/sneha/Documents/git/geolocation_attributes/raw/yfcc100m_dataset-3')
yli_geo_train_path = ('/home/sneha/Documents/git/geolocation_attributes/raw/mediaeval2016_placing_train_photo')

# consider top MAX images from yfcc100m_dataset-0
sample = []
sample_url = []
# images from top MAX from YFCC100M included in YLI-GEO
dataset = []

MAX = 2000000
count = 0
with open(yfcc100m_path) as tsv:
    reader = csv.reader(tsv)
    for row in reader:
        if(count<MAX):
            # only data points that have url available
            if len(row[0].split('\t')) >=14 :
                sample.append(int(row[0].split('\t')[0]))
                sample_url.append(row[0].split('\t')[14])
            count = count + 1
        else:
            break

c = 0

with open(yli_geo_train_path) as tsv:
    reader = csv.reader(tsv, delimiter='\t')
    for row in reader:
        # if(binary_search(sample, int(row[0].split('\t')[0]))):
        # if image is present in the downloaded yfc100m sample
        if(int(row[0]) in sample):
            print c
            # check if an image is available at the corresponding url
            try:
                # keep the example only if country and town present
                places = row[4].split(',')
                town = 0
                country = 0
                for place in places:
                    if 'Town' in place:
                        town = 1
                    if 'Country' in place:
                        country = 1
                if(country == 1 and town == 1):
                    url = sample_url[sample.index(int(row[0]))]
                    image = urllib.URLopener()
                    image.retrieve(url, 'data/train/images-data-3/'+row[0]+'.jpg')
                    im = Image.open('data/train/images-data-3/'+row[0]+'.jpg')
                    im = im.resize(size=(200, 200))
                    im.save('data/train/images-data-3/'+row[0]+'.jpg')
                    row.append(sample_url[sample.index(int(row[0]))])
                    dataset.append(row)
                with open('data/train/dataset-3', 'a') as f:
                    f.write("%s\n" % row)
            except IOError:
                # if image not available, don't append to the dataset
                continue
        c = c+1


print len(dataset)
