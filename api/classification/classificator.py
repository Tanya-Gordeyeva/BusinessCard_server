import json
from math import log

from api.classification.train import Classifier, get_features
from natasha import AddressExtractor
import re
import numpy as np


class Person:
    name = ''
    surname = ''
    fathername = ''
    phone = ''
    address = ''
    email = ''
    position = ''
    organization = ''
    image = ''
    site = ''
    notes = ''


def classify(classifier, feats):
    classes, prob = classifier
    name = [log(classes['Имя']) + sum(log(prob.get(('Имя', feat), 10 ** (-7))) for feat in feats)]
    surname = [log(classes['Фамилия']) + sum(log(prob.get(('Фамилия', feat), 10 ** (-7))) for feat in feats)]
    fathername = [log(classes['Отчество']) + sum(log(prob.get(('Отчество', feat), 10 ** (-7))) for feat in feats)]
    return max(classes.keys(),
               key=lambda cl: log(classes[cl]) + sum(
                   log(prob.get((cl, feat), 10 ** (-7))) for feat in feats)), max(name, surname, fathername)


def textClassificator(text):
    person = Person()

    phones = re.findall(r'([+(]?[1-9][0-9 .\-()]{8,}[0-9])|([0-9.\-()]{7,})', text)
    person.phone = [value for value in np.array(phones).flatten() if value]

    person.email = re.findall(r'([-._a-z0-9]+@(?:[a-z0-9][-a-z0-9]+\.)+[a-z]{2,6})', text)

    symb = re.compile(r'(моб|mob|mail|fax|факс|tel|тел\.|телефон|сайт|site)', re.IGNORECASE)
    symbols = symb.findall(text)
    t = text.split()

    site = re.compile(r'(http:\s*//www.\w+.\w+)|(www.\w+.\w+)', re.IGNORECASE)
    site_list = site.findall(text)
    person.site = [re.sub(" ","",value) for value in np.array(site_list).flatten() if value]

    for word in t:
        for substr in symbols:
            if substr in word:
                text = text.replace(word, "")

    for i in range(len(site_list)):
        text = text.replace(np.array(site_list).flatten()[i], "")

    for i in range(len(person.phone)):
        text = text.replace(person.phone[i], "")

    for i in range(len(person.email)):
        text = text.replace(person.email[i], "")

    text = text.replace('\n', " ")
    extractor = AddressExtractor()

    matches = extractor(text)
    facts = [_.fact.as_json for _ in matches]
    if facts:
        address = facts[0]['parts']
        address_list = ''

        for i in range(len(address)):
            keys = reversed(address[i].keys())
            for key in keys:
                address_list += '' + address[i][key] + ' '
                text = re.sub(r'\s' + address[i][key] + '\s|,', " ", text)
        person.address = address_list

    symb = re.compile(r'(г\.|город|ул\.|улица|стр\.|строение|д\.|дом|офис|помещение|к\.|корпус)')
    symbols = symb.findall(text)
    t = text.split()

    for word in t:
        for substr in symbols:
            if substr in word:
                text = text.replace(word, "")

    words = text.split()
    nlf_cl = []
    if len(Classifier.classifier[0]):
        pos_n = 0
        pos_s = 0
        pos_f = 0
        for word in words:
            if len(word) > 1:
                nlf_cl.append(classify(Classifier.classifier, get_features(word)))
        max_name = -np.inf
        max_surname = -np.inf
        max_fathername = -np.inf
        for idx in range(len(nlf_cl)):
            key, value = nlf_cl[idx]
            if key == 'Имя':
                if max_name < value[0]:
                    max_name = value[0]
                    pos_n = idx
            if key == 'Фамилия':
                if max_surname < value[0]:
                    max_surname = value[0]
                    pos_s = idx
            if key == 'Отчество':
                if max_fathername < value[0]:
                    max_fathername = value[0]
                    pos_f = idx
        person.name = words[pos_n]
        person.surname = words[pos_s]
        person.fathername = words[pos_f]
        text = re.sub(r'\s|' + person.name + '\s|,', " ", text)
        text = re.sub(r'\s|' + person.surname + '\s|,', " ", text)
        text = re.sub(r'\s|' + person.fathername + '\s|,', " ", text)
        person.notes = text
    else:
        return []
    return json.dumps(person, default=lambda o: o.__dict__)
