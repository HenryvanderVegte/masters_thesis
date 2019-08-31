import librosa, librosa.display
from tqdm import tqdm
import pickle
from utils.rnn_utils import *
import torch.utils.data as utils
from torch.nn.utils.rnn import pad_sequence
from models import CNNBLSTM
from utils.experiments_util import *

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

#length: 4 seconds
example_path = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP//features//audio//wavs//Ses01F_impro01//Ses01F_impro01_F005.wav"
EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//"
wavs_path = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP//features//audio//wavs"
metadata = read_tsv_metadata("C://Users//Henry//Desktop//Masterarbeit//IEMOCAP//labels.tsv")

n_fft_ms = 25
hop_length_ms = 10
n_mels = 40

def create_sequence_dataset_from_metadata(metadata, class_groups, set, max_seq_length = None):
    """
    Loads sequence data into a TensorDataset,
    if max_seq_length is set, sequences longer than seq_length will be cut.
    sequences shorter than max_seq_length will be zero-padded
    :param feature_dict:
    :param label_dict:
    :param max_seq_length:
    :return:
    """
    fl = []
    labels = []
    lengths = []
    ids = []
    for instance in tqdm(metadata):
        if instance["Label"] not in class_groups or instance["Set"] != set:
            continue

        label = class_groups[instance["Label"]]
        audio_path = os.path.join(wavs_path, instance["Audio"])

        y, sr = librosa.load(audio_path)

        n_fft = int(sr / (1000 / n_fft_ms))
        hop_length = int(sr / (1000 / hop_length_ms))

        features = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels).T

        features = librosa.power_to_db(features**2, ref=np.max)

        if max_seq_length is None:
            fl.append(torch.stack([torch.Tensor(i) for i in features]))
        else:
            fl.append(torch.stack([torch.Tensor(i) for i in features[:max_seq_length, :]]))

        length = features.shape[0] if max_seq_length is None else min(features.shape[0], max_seq_length)
        lengths.append(length)
        labels.append(label)
        ids.append(int(instance["Id"]))

    labels = np.array(labels).reshape(-1,1)
    lengths = np.array(lengths).reshape(-1,1)
    ids = np.array(ids).reshape(-1,1)
    padded_features = pad_sequence(fl)

    labels = torch.stack([torch.Tensor(i) for i in labels])
    lengths = torch.stack([torch.Tensor(i) for i in lengths])
    ids = torch.stack([torch.Tensor(i) for i in ids])

    padded_features = torch.transpose(padded_features, 0, 1)
    dataset = utils.TensorDataset(padded_features, labels, lengths, ids)

    return dataset

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_audio_lstm", use_timestamp=True)

'''
train_dataset = create_sequence_dataset_from_metadata(metadata, class_groups, "train")
dev_dataset = create_sequence_dataset_from_metadata(metadata, class_groups, "dev")


with open("E://masters_thesis//dev.dataset", 'wb') as dev_dataset_path:
    pickle.dump(dev_dataset, dev_dataset_path)

with open("E://masters_thesis//train.dataset", 'wb') as train_dataset_path:
    pickle.dump(train_dataset, train_dataset_path)
'''

with open("E://masters_thesis//dev.dataset", 'rb') as dev_dataset_path:
    dev_dataset = pickle.load(dev_dataset_path)

with open("E://masters_thesis//train.dataset", 'rb') as train_dataset_path:
    train_dataset = pickle.load(train_dataset_path)


id_to_name = {}
for m in metadata:
    id_to_name[int(m["Id"])] = m["Name"]

params = {
    "max_sequence_length": 50,
    "batch_size": 4,
    "hidden_size": 256,
    "drop_prob": 0.2,
    "fully_connected_drop_prob": 0.4,
    "layers": 2,
    "epochs": 1000,
    "log_x_epochs": 5,
}

params["labels_size"] = len(set(list(class_groups.values())))
params["embedding_dim"] = dev_dataset.tensors[0][0].size()[1]

model = CNNBLSTM.CNNBLSTM(params)

train(train_dataset, dev_dataset, id_to_name, experiment_dir, model, logger, params)