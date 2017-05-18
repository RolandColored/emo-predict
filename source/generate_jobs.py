from config.pipelineconfig import PipelineConfig

print('#!/bin/sh')


datasets = ['foxnews+nytimes+theguardian', 'bild+spiegelonline+ihre.sz']
features = [func for func in dir(PipelineConfig) if callable(getattr(PipelineConfig, func))
            and 'tfidf' not in func
            and not func.startswith('_')]
regressors = ['Baseline', 'Ridge']


for dataset in datasets:
    for feature in features:
        for reg_name in regressors:
            print('qsub -v datasource=' + dataset + ',feature=' + feature + ',regressor=' + reg_name + ' emopredict.job')
