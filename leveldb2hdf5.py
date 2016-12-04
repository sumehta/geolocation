import caffe
import leveldb
import numpy as np
import h5py
from caffe.proto import caffe_pb2
db = leveldb.LevelDB('examples/_temp/features')
datum = caffe_pb2.Datum()

arr = np.empty((11126, 4096), dtype=float)
i = 0
for key, value in db.RangeIter():
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)
    arr[i, :] = data.reshape(4096)
    i = i+1
    print i
with h5py.File('data.h5', 'w') as h5:
    h5.create_dataset('image-features', data=arr)

    # image = np.transpose(data, (1,2,0))

    # np.save('feature.txt',image)
