import csv

with open('red.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    row_count = 0
    for row in csv_reader:
        if row_count > 0:
            print([float(i) for i in row[21:25]])
        row_count += 1
