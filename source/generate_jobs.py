print('#!/bin/sh')


datasets = ['foxnews+nytimes+theguardian', 'bild+spiegelonline+ihre.sz']
#features = [func for func in dir(PipelineConfig) if callable(getattr(PipelineConfig, func)) and not func.startswith('_')]
features = ['text_bigram_bow_100', 'text_bigram_bow_1000', 'text_bigram_bow_250', 'text_bigram_bow_500', 'text_bigram_lemmatized_bow_100', 'text_bigram_lemmatized_bow_1000', 'text_bigram_lemmatized_bow_250', 'text_bigram_lemmatized_bow_500', 'text_bigram_stemmed_bow_100', 'text_bigram_stemmed_bow_1000', 'text_bigram_stemmed_bow_250', 'text_bigram_stemmed_bow_500', 'text_bow_100', 'text_bow_1000', 'text_bow_250', 'text_bow_500', 'text_lemmatized_bow_100', 'text_lemmatized_bow_1000', 'text_lemmatized_bow_250', 'text_lemmatized_bow_500', 'text_stemmed_bow_100', 'text_stemmed_bow_1000', 'text_stemmed_bow_250', 'text_stemmed_bow_500']
regressors = ['Baseline', 'Ridge']


for dataset in datasets:
    for feature in features:
        for reg_name in regressors:
            print('qsub -v datasource=' + dataset + ',feature=' + feature + ',regressor=' + reg_name + ' emopredict.job')
