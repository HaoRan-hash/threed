from nni.experiment import Experiment


search_space = {
    'coarse_seg_loss_weight1': {'_type': 'uniform', '_value': [0.1, 1.0]},
    'coarse_seg_loss_weight2': {'_type': 'uniform', '_value': [0.1, 1.0]},
}

experiment = Experiment('local')

experiment.config.trial_command = 'python seg/memorynet_seg_v2.py'
experiment.config.trial_code_directory = '.'
experiment.config.trial_concurrency = 1

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.run(8081)