from nni.experiment import Experiment


search_space = {
    'contrast_loss_weight1': {'_type': 'uniform', '_value': [0.001, 0.05]},
    'contrast_loss_weight2': {'_type': 'uniform', '_value': [0.001, 0.05]},
}

experiment = Experiment('local')

experiment.config.trial_command = 'python seg/memorynet_seg_v3.py'
experiment.config.trial_code_directory = '.'
experiment.config.trial_concurrency = 1

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.run(8080)