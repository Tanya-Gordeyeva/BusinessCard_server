from collections import defaultdict
import numpy as np
import random
from api.models import Names,Fathernames,Lastnames

class Classifier(object):
    classifier = defaultdict(lambda: 0), defaultdict(lambda: 0)

def trainD():
    names = Names.objects.all()
    lastnames = Lastnames.objects.all()
    fathernames = Fathernames.objects.all()
    data_fnames = list(fathernames)
    data_names = list(names)
    data_surnames = list(lastnames)
    samples = readData(data_fnames,data_names,data_surnames)
    features = [(get_features(samples[0][i]), samples[1][i]) for i in range(samples.shape[1])]
    Classifier.classifier = train(features)
    return True

def get_features(sample):
    if (len(sample)>2):
        return (
        'll: %s' % sample[-1].lower(),
        'ml: %s' % sample[-2].lower(),
        'fl: %s' % sample[0],
        'sl: %s' % sample[-3].lower()
        )
    else:
        return (
        'll: %s' % sample[-1].lower(),
        'ml: %s' % sample[-2].lower(),
        'fl: %s' % sample[0]
        )

def train(samples):
    classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for feats, label in samples:
        classes[label] += 1
        for feat in feats:
            freq[label, feat] += 1
    for label, feat in freq:
        freq[label, feat] /= classes[label]
    for c in classes:
        classes[c] /= len(samples)
    return classes, freq

def readData(data_fnames,data_names,data_surnames):
    dataset = []
    labels = []
    for i in range(1500):
        dataset.append(data_fnames[random.randint(0, len(data_fnames) - 1)].fathername)
        labels.append('Отчество')
        dataset.append(data_names[random.randint(0, len(data_names) - 1)].name)
        labels.append('Имя')
        dataset.append(data_surnames[random.randint(0, len(data_surnames) - 1)].lastname)
        labels.append('Фамилия')
    dataset = np.array(dataset)
    labels = np.array(labels)
    samples = np.vstack([dataset, labels])
    return samples