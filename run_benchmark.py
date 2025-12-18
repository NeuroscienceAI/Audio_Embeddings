import sys
from plumbum import local, FG
from plumbum.commands.processes import ProcessExecutionError

python = local['python']
local['mkdir']['-p', 'logs']()

MODELS = [
    
    ###1 Torchaudio & mfcc inputs, non-DL
    # ('lr', 'lr', '{"C": [0.1, 1.0, 10.0]}'),
    # ('svm', 'svm', '{"C": [0.1, 1.0, 10.0]}'),
    # ('decisiontree', 'decisiontree', '{"max_depth": [None, 5, 10, 20, 30]}'),
        ###('rf', 'rf', '{"n_estimators": [3, 5, 10, 50, 100, 200]}'),
    # ('gbdt', 'gbdt', '{"n_estimators": [10, 50, 100, 200]}'),
    # ('xgboost', 'xgboost', '{"n_estimators": [10, 50, 100, 200]}'),
    
    # ('alexnet-pretrained', 'alexnet-pretrained', ''),
    
    ###2 CNN & spectrogram inputs, random or ImageNet pretrained
    # ('resnet18', 'resnet18', ''),
    # ('resnet18-pretrained', 'resnet18-pretrained', ''),
    # ('resnet50', 'resnet50', ''),
    # ('resnet50-pretrained', 'resnet50-pretrained', ''),
    ('resnet152', 'resnet152', ''),
    # ('resnet152-pretrained', 'resnet152-pretrained', ''),
    
    ### ('convnext-pretrained', 'convnext-pretrained', ''),
    
    # ('swin', 'swin', ''),
    
    # ('swin-pretrained', 'swin-pretrained', ''),
    
    ###3 CNN & spectrogram inputs, YouTube all audio pretrained
    # ('vggish', 'vggish', ''), 
    
    ###4-1 Transformer & waveform inputs, bio-audio pretrained
    # ('aves', 'aves', ''),
    
    ###4-1 Transformer & waveform inputs, all-audio pretrained
    # ('aves-all', 'aves-all', ''),    
    
    ###5 SWINTransformer & spectrogram inputs, audio+text pretrained
    ### ('clap', 'clap', ''),
    
    ###6 SWINTransformer & spectrogram inputs, bio-audio+text pretrained
    ### ('biolingual', 'biolingual', ''),
]

TASKS = [
    ('classification', 'watkins'),
    ('classification', 'bats'),
    ('classification', 'dogs'),
    ('classification', 'cbi'),
    ('classification', 'humbugdb'),
    ('detection', 'dcase'),
    ('detection', 'enabirds'),
    ('detection', 'hiceas'),
    ('detection', 'rfcx'),
    ('detection', 'hainan-gibbons'),
]

for model_name, model_type, model_params in MODELS:
    for task, dataset in TASKS:
        print(f'Running {dataset}-{model_name}', file=sys.stderr)
        log_path = f'logs/{dataset}-{model_name}'
        try:
            if model_type in ['lr','svm','neighbors','naive_bayes','decisiontree','rf','gbdt','xgboost']:
                python[
                    'scripts/evaluate_1GPU.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--params', model_params,
                    '--log-path', log_path,
                    '--num-workers', '1'] & FG
            else:
                python[
                    'scripts/evaluate_1GPU.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--batch-size', '32',
                    '--epochs', '1',
                    '--lrs', '[1e-5]', ### [1e-5, 5e-5, 1e-4, 5e-4]
                    '--log-path', log_path,
                    '--num-workers', '1'] & FG
        except ProcessExecutionError as e:
            print(e)
