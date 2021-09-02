with open('citeseer.content','r+') as fb:
    with open('citeseer.label', 'w+') as fb1:
        for line in fb:
            line1 = line.split()
            touple = []
            touple.append(line1[0])
            touple.append(',')
            touple.append(line1[-1])
            fb1.writelines(touple)
            fb1.writelines('\n')
            print(line)