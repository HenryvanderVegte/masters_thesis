import os, datetime
import json

def create_experiment(experiments_dir, label_to_id, description):
    experiment_dir = os.path.join(experiments_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(experiment_dir)
    logfile = os.path.join(experiment_dir, 'log.txt')
    with open(logfile, 'w') as d:
        d.write("Experiment:\n")
        d.write(description+ "\n\n")
        d.write("Labels"+ "\n")
        d.write(json.dumps(label_to_id))
        d.write("\n\n")
    return experiment_dir, logfile
