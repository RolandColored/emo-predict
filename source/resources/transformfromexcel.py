import csv

with open('NRC-emotion-lexicon-de.csv') as csv_file:
    with open('NRC-emotion-lexicon-de.txt', mode='w') as out_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            out_file.write(row['word'] + '\tanger\t' + row['anger'] + '\n')
            out_file.write(row['word'] + '\tanticipation\t' + row['anticipation'] + '\n')
            out_file.write(row['word'] + '\tdisgust\t' + row['disgust'] + '\n')
            out_file.write(row['word'] + '\tfear\t' + row['fear'] + '\n')
            out_file.write(row['word'] + '\tjoy\t' + row['joy'] + '\n')
            out_file.write(row['word'] + '\tnegative\t' + row['negative'] + '\n')
            out_file.write(row['word'] + '\tpositive\t' + row['positive'] + '\n')
            out_file.write(row['word'] + '\tsadness\t' + row['sadness'] + '\n')
            out_file.write(row['word'] + '\tsurprise\t' + row['surprise'] + '\n')
            out_file.write(row['word'] + '\ttrust\t' + row['trust'] + '\n')
