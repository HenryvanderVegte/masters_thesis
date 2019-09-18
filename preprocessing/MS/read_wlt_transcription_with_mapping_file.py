import json

mapping_file = 'C://Users//Henry//Desktop//datasets//MS//sentimentdataset_transcriptions//sentimentdataset//train//5//mapping.txt'
json_file = 'C://Users//Henry//Desktop//datasets//MS//sentimentdataset_transcriptions//sentimentdataset//train//5//unidec_out.json'
output_file  = "C://Users//Henry//Desktop//datasets//MS//sentiment dataset//transcriptions//train5.txt"

mapping_lines = open(mapping_file).read().splitlines()

filenamemapping = {}
for line in mapping_lines:
    split = line.split('\t')
    name = split[0]
    audiofile = '/'.join(split[2].split('/')[2:])
    filenamemapping[split[0]] = audiofile



with open(json_file, 'r') as f:
    json_obj = json.load(f)
    results = json_obj['Results']

    out_str = ''

    for result in results:
        file_err = False
        internalname = result['Audio']
        internalnamesplit = internalname.split('.')
        if internalnamesplit[2] != '0':
            continue

        internalnamefull = internalnamesplit[0] + '.' + internalnamesplit[1] + '.' + internalnamesplit[3]
        line = filenamemapping[internalnamefull]

        for sentence in result['Sentences']:
            if 'NBestList' not in sentence or len(sentence['NBestList']) == 0:
                file_err = True
                print("File error: " + internalname)
                continue

            for nbest in sentence['NBestList']:
                for word in nbest['WordList']:
                    if word['Display'].startswith('!sent'):
                        continue
                    wordDisplay = word['Display']
                    start = word['StartTime']
                    end = word['EndTime']
                    diff = end - start
                    line += '\t' + wordDisplay + '\t' + str(float(start) / 1000) + ';' +  str(float(diff) / 1000)

        out_str += line + '\n'

    with open(output_file, 'w') as w:
        w.write(out_str)