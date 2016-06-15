# coding:utf-8

import os
import io
import urllib2
from PIL import Image
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from pymongo import MongoClient


class Detecter():

    def __init__(self, host, port):
        self.STANDARD_SIZE = (300, 167)
        self.connection = MongoClient(host, port)

    def img_to_matrix(self, url, verbose=False):
        fd = urllib2.urlopen(url)
        image_file = io.BytesIO(fd.read())
        img = Image.open(image_file)
        if verbose:
            print 'changing size from %s to %s' % (str(img.size), str(self.STANDARD_SIZE))
        img = img.resize(self.STANDARD_SIZE)
        imgArray = np.asarray(img)
        return imgArray  # imgArray.shape = (167 x 300 x 3)

    def flatten_image(self, img):
        s = img.shape[0] * img.shape[1] * img.shape[2]
        img_wide = img.reshape(1, s)
        return img_wide[0]

    def detect(self, imageURLs, params):

        array = []
        for param in params:
            img = self.img_to_matrix(param['imageURL'])
            data = self.flatten_image(img)
            array.append(data)
        array = np.array(array)

        pca = RandomizedPCA(n_components=5)
        n_data = pca.fit_transform(array)

        clf = joblib.load('src/resource/models/model.pkl')
        result = clf.predict(n_data).tolist()

        for param, r in zip(params, result):
            raw_img = urllib2.urlopen(param['imageURL']).read()
            if r == 1:
                cntr = len([i for i in os.listdir("test/images/rain/") if 'rain' in i]) + 1
                path = "static/images/rain_" + str(cntr) + '.jpg'
                f = open(path, 'wb')
                f.write(raw_img)
                f.close()
                # イベント情報作成
                when = {'type': 'timestamp', 'time':param['time']}
                where = { "type": "Point", "coordinates": [param['longitude'], param['latitude']]}
                what = {'topic': {'value':u'雨'}, 'tweet': param['value']}
                who = [{"type": "url", "value": param['imageURL']},
                       {"value": "evwh <evwh@evwh.com>", "type": "author"}]
                event = {'observation':{'what': what, 'when': when, 'where': where, 'who': who}}
                self.connection['event']['TwitterImageRainSensor'].insert(event)


if __name__ == '__main__':

    import src.harvest.bing_harvester as harvest
    harvester = harvest.BingHarvester()
    imageURLs = harvester.get_urls('雨')
    detect = Detecter()
    detect.detect(imageURLs)