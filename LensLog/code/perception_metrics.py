"""Independently tested DEVELOPMENT proxies, not official Samsung metrics."""
import numpy as np


def ranking_f1(human,predicted,top_ratio=.15):
    human=np.asarray(human,dtype=float);predicted=np.asarray(predicted,dtype=float)
    if human.ndim!=1 or human.shape!=predicted.shape or not len(human) or not np.isfinite(human).all() or not np.isfinite(predicted).all() or np.any((human<0)|(human>1)) or np.any((predicted<0)|(predicted>1)):
        raise ValueError('Ranking F1 requires aligned nonempty finite [0,1] vectors.')
    if not np.isfinite(top_ratio) or not 0<top_ratio<=1:raise ValueError('top_ratio must be in (0,1].')
    k=max(1,int(np.ceil(top_ratio*len(human))))
    truth=set(np.argsort(-human,kind='stable')[:k]);selection=set(np.argsort(-predicted,kind='stable')[:k])
    value=len(truth&selection)/k
    return {'metric':'sample-grid-importance-ranking-F1-proxy-not-official','precision':value,'recall':value,'f1':value,'positive_samples':k,'samples':len(human),'ties':'earlier sample first','budget_enforced':False}


def category_metrics(labels,truth,predicted):
    labels=tuple(labels)
    if not labels or len(set(labels))!=len(labels) or len(truth)!=len(predicted) or not len(truth):
        raise ValueError('Need unique categories and equal nonempty label sequences.')
    matrix=np.zeros((len(labels),len(labels)),dtype=int);index={x:i for i,x in enumerate(labels)}
    for actual,guess in zip(truth,predicted):
        if actual not in index or guess not in index:raise ValueError('Category outside approved evaluation vocabulary.')
        matrix[index[actual],index[guess]]+=1
    counts=matrix.sum(axis=1)
    return {'accuracy':float(np.trace(matrix)/len(truth)),'sample_count':len(truth),'labels':list(labels),'confusion_matrix':matrix.tolist(),'per_category':{label:{'count':int(counts[i]),'accuracy':float(matrix[i,i]/counts[i]) if counts[i] else None} for i,label in enumerate(labels)}}
