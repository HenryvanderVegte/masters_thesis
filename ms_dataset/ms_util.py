def read_ms_tsv_metadata(dataset):
    """
    Reads given .tsv metadata file and returns every instance in a list of dictionaries
    :param dataset:
    :return: list of dictionaries, each representing one instance
    """
    d = open(dataset).read().splitlines()
    headers = d[0].split("\t")
    ret = []
    for row in d[1:]:
        row = row.split("\t")

        instance = {headers[i]: row[i] for i in range(len(row))}
        id = instance['mPhraseId']
        instance['Name'] = instance['mRawDataId'] + '/' + id.zfill(4) + '.m4a'
        instance['Label'] = instance['Rating']
        ret.append(instance)
    return ret

