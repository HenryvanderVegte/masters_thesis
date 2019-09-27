import os
import json

def read_transcriptions_from_json(input_folder, output_file):
    """
    Takes a folder of transcriptions in .json format produced by the Microsoft Custom Speech Service
    (https://westus.cris.ai/Home/CustomSpeech) and reads out the transcription and word level timestamps,
    writes all into a single file.
    :param input_folder:
    :param output_file:
    :return:
    """
    out_str = ''
    for r, d, f in os.walk(input_folder):
        for file in f:
            file_path = os.path.join(r, file)
            with open(file_path, 'r') as f:
                file_err = False

                json_obj = json.load(f)
                file_name = '/'.join(json_obj['AudioFileResults'][0]['AudioFileName'].split('/')[2:])

                segment_results = json_obj['AudioFileResults'][0]['SegmentResults']

                if len(segment_results) == 0:
                    print("0 segments: " + file_name + '\t' + file_path)
                    continue

                line = file_name

                for segment in segment_results:
                    if segment['ChannelNumber'] != '0':
                        continue
                    nbests = segment['NBest']
                    if len(nbests) != 1:
                        print("NBest length: " + str(len(nbests)) + ' ' + file_path)
                        file_err = True
                        continue
                    nbest = nbests[0]
                    words = nbest['Words']
                    if len(words) == 0:
                        print("Words length: " + str(len(words))+ ' ' + file_path)
                        file_err = True
                        continue

                    for word in words:
                        line += '\t' + word['Word'] + '\t' + str(word['OffsetInSeconds']) + ';' + str(word['DurationInSeconds'])

                if not file_err:
                    out_str += line + '\n'
                else:
                    print('Error in ' + file_name)

    with open(output_file, 'w') as w:
        w.write(out_str)

