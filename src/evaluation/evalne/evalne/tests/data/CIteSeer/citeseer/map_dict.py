import json

def relabel_graph(dict1):
    with open('citeseer.cites', 'r') as f1:
        with open('citeseer.cites1', 'w+') as f2:
            for line in f1:
                touple = []
                line1 = line.split()

                touple.append(dict1[line1[0]])
                touple.append(',')
                touple.append(dict1[line1[1]])
                touple.append('\n')
                f2.writelines(touple)

def relabel_label(dict1, dict2):
    with open('citeseer.label', 'r') as f1:
        with open('citeseer_label.txt', 'w+') as f2:
            for line in f1:
                touple = []
                line1 = line.split(',')

                touple.append(dict1[line1[0]])
                touple.append(',')
                touple.append(dict2[line1[1].strip()])
                touple.append('\n')
                f2.writelines(touple)

# with open('citeseer.cites', 'r+')as fp:
#     list1 = []
#     for line in fp:
#         nodes = line.split()
#         list1.extend(nodes)
#     list2 = list(set(list1))
#     dict1 = {}
#     for idx,item in enumerate(list2):
#         dict1[str(item)] = str(idx)
#
#     with open('mapdict.txt', 'w+') as fp1:
#         json.dump(dict1, fp1)
def map_label():
    with open('citeseer.label', 'r+')as fp:
        list1 = []
        for line in fp:
            nodes = line.split(',')
            list1.append(nodes[1].strip())
        list2 = list(set(list1))
        dict1 = {}
        for idx,item in enumerate(list2):
            dict1[str(item)] = str(idx)

        with open('map_label.txt', 'w+') as fp1:
            json.dump(dict1, fp1)

with open('mapdict.txt', 'r')as f4:
    with open('map_label.txt', 'r')as f5:
        dict1 = json.load(f4)
        dict2 = json.load(f5)
        relabel_label(dict1, dict2)

