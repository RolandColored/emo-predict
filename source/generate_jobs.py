from config.pipelineconfig import PipelineConfig


datasets = ['foxnews+nytimes+theguardian']#, 'bild+spiegelonline+ihre.sz']
features = [func for func in dir(PipelineConfig) if callable(getattr(PipelineConfig, func))
            and 'tfidf' not in func
            and not func.startswith('_')]
regressors = ['Ridge']

i = 0

for dataset in datasets:
    for feature in features:
        for reg_name in regressors:
            print("job[" + str(i) + "]='python3 /home/rokah100/emo-predict/source/emopredict.py " + dataset + " " + feature + " " + reg_name + " /home/rokah100/results'")
            i += 1

print('run=${job[$PBS_ARRAY_INDEX]}')
print('eval $run')
