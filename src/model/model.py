# coding:utf-8

import os
from PIL import Image
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC


class Modeling():

    def __init__(self):
        self.STANDARD_SIZE = (300, 167)

    def img_to_matrix(self, filename, verbose=False):
        img = Image.open(filename)
        if verbose:
            print 'changing size from %s to %s' % (str(img.size), str(self.STANDARD_SIZE))
        img = img.resize(self.STANDARD_SIZE)
        imgArray = np.asarray(img)
        return imgArray 

    def flatten_image(self, img):
        s = img.shape[0] * img.shape[1] * img.shape[2]
        img_wide = img.reshape(1, s)
        return img_wide[0]

    def execute(self):
        img_dir = 'src/resource/images/'
        images = [img_dir + f for f in os.listdir(img_dir)]
        labels = ['rain' if 'rain' in f.split('/')[-1] else 'sunny' for f in images]

        data = []
        for image in images:
            img = self.img_to_matrix(image)
            img = self.flatten_image(img)
            data.append(img)

        data = np.array(data)

        is_train = np.random.uniform(0, 1, len(data)) <= 0.7
        y = np.where(np.array(labels) == 'rain', 1, 0)

        train_x, train_y = data[is_train], y[is_train]

        # plot in 2 dimensions
        pca = RandomizedPCA(n_components=2)
        X = pca.fit_transform(data)
        df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1],
                           "label": np.where(y == 1, 'rain', 'sunny')})
        colors = ['red', 'yellow']
        for label, color in zip(df['label'].unique(), colors):
            mask = df['label'] == label
            pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)

        pl.legend()
        pl.savefig('src/resource/feature_image/pca_feature.png')

        # training a classifier
        pca = RandomizedPCA(n_components=5)
        train_x = pca.fit_transform(train_x)

        svm = LinearSVC(C=1.0)
        svm.fit(train_x, train_y)
        joblib.dump(svm, 'src/resource/models/model.pkl')

        # evaluating the model
        test_x, test_y = data[is_train == False], y[is_train == False]
        test_x = pca.transform(test_x)
        print pd.crosstab(test_y, svm.predict(test_x),
                          rownames=['Actual'], colnames=['Predicted'])


if __name__ == '__main__':
    detector = Modeling()
    detector.execute()