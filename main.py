import numpy as np
import cv2
import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths


# V11PO03
# #V10PO02
# Command Line argument parser

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-t', '--trials', type=int, default=5, help='# of trials to run')
# ap.add_argument('-d', '-dataset')
# ap.add_argument('-t', '--trials', type=int, default=5)

args = vars(ap.parse_args())

# define the training and testing paths
training_path = os.path.sep.join([args['dataset'], 'training'])
testing_path = os.path.sep.join([args['dataset'], 'testing'])
# training_path = os.path.sep.join([ 'training'])
# testing_path = os.path.sep.join(['testing'])

print('[INFO] Loading data...')
# load the data into training and testing inputs and outputs
(trainX, trainY) = load_split(training_path)
(testX, testY) = load_split(testing_path)

# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize our trials dictionary
trials = {}

# start the trials
for i in range(0, args['trials']):

    # train the model
    print('Training the model {} of {}...'.format(i + 1, args['trials']))

    model = RandomForestClassifier(n_estimators=100)
    model.fit(trainX, trainY)

    # making predictions on the test dataset
    predictions = model.predict(testX)
    metrics = {}

    # Compute the confusion_matrix to measure accuracy
    cm = confusion_matrix(testY, predictions).flatten()
    (TrueNeg, FalsePos, FalseNeg, TruePos) = cm
    metrics['accuracy'] = (TruePos + TrueNeg) / float(cm.sum())
    metrics['Sensitivity'] = TruePos / float(TruePos + FalseNeg)
    metrics['Specificity'] = TrueNeg / float(TrueNeg + FalsePos)

    for (k, v) in metrics.items():
        l = trials.get(k, [])
        l.append(v)
        trials[k] = l

for metric in ('accuracy', 'Sensitivity', 'Specificity'):
    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)

    # print the computed metric
    print(metric)
    print('=' * len(metric))
    print('u={:.4f} , o={:.4f}'.format(mean, std))
    print("")

testing_paths = list(paths.list_images(testing_path))
idx = np.arange(0, len(testing_paths))
idx = np.random.choice(idx, size=(25,), replace=False)
images = []

for i in idx:
    image = cv2.imread(testing_paths[i])
    output = image.copy()
    output = cv2.resize(output, (128, 128))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    features = quantify_images(image)
    pre = model.predict([features])
    label = le.inverse_transform(pre)[0]

    color = (0, 255, 0) if label == 'healthy' else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    images.append(output)

montage = build_montages(images, (128, 128), (5, 5))[0]

cv2.imshow('Output', montage)
cv2.waitKey(0)

def quantify_images(im):
    fea = feature.hog(im, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                      transform_sqrt=True, block_norm='L1')
    return fea


def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imager in imagePaths:
        point = imager.split(os.path.sep)[-2]

        im1 = cv2.imread(imager)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im1 = cv2.resize(im1, (200, 200))

        im1 = cv2.threshold(im1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        fea1 = quantify_images(im1)

        data.append(fea1)
        labels.append(point)

    return np.array(data), np.array(labels)


