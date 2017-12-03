import csv

def libsvmdump(filename, data, labels):
    file = open(filename, 'w')
    # writer = csv.writer(file)

    # for i in len(data):
    #     d = data[i]
    #     l = labels[i]
    #     row = [l]

    #     for 
    #     writer.writerow(line)
    # file.close()

def csvdump(filename, data):
    file = open(filename, 'w')
    writer = csv.writer(file)

    for line in data:
        writer.wri
        writer.writerow(line)
    file.close()