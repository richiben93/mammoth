import pandas as pd
import os
import numpy as np

def method_to_df(apath, setting, dataset, method):
    last = ' '.join([setting, dataset, method])
    with open(apath + '%s/%s/' % (setting, dataset) + method +'/logs.pyd', 'r') as f:
        lines = f.read().splitlines()
    o = pd.DataFrame.from_records([eval(x.replace('nan', 'float(\'nan\')')) for x in lines])
    
    o = o.rename(columns={'cs_size':'buffer_size'})
    if 'buffer_size' not in o:
        o['buffer_size'] = 0
    if 'job_number' in o.columns:
        del o['job_number']
    o['method'] = method
    return o


req_rep = 1
print_args= False
display_pre = False
howmany_pre = 1
print_other_metrics = False
t_distinct_cols = ['buffer_size', 'method', 'n_epochs']
ignore_cols = []
excluded_methods = []
included_methods = []
excluded_datasets = []

sec_respath = None

# PAMI
respath = '/home/aba/memes/mammothshap/data/results/'
sec_respath = None #'/home/mbosc/srv-nas/mammoth-master/results/'
third_respath = None #'/home/mbosc/phd/pami/results_buzz/'

respaths = [respath, sec_respath, third_respath]
islist = lambda col: any([type(x) == list for x in col.values])

def aggf(x):
#     if len(x) > 2:
#         return x.sort_values()[1:-1].mean()
#     else:
        return x.mean()

# -------------------

for setting in ['task-il']:
    if not os.path.isdir(respath + setting): continue
        
    for dataset in [x for x in set(os.listdir(respath + setting) + (os.listdir(sec_respath + setting) if sec_respath else [])) if x not in excluded_datasets]:
        
        print(dataset)
        
        # -------------------------- READ LOG FILES INTO A SINGLE DATAFRAME ----------------------------
        methods = None
        for apath in [x for x in respaths if x is not None]:
            if dataset in excluded_datasets: continue
            
            if not os.path.isdir(apath + '%s/%s/' % (setting, dataset)): continue;
            
            for method in os.listdir(apath + '%s/%s/' % (setting, dataset)):
                if method.startswith('__'): continue;
                if method in excluded_methods: continue;
                if len(included_methods) > 0 and method not in included_methods: continue;
                if not os.path.isfile(apath + '%s/%s/' % (setting, dataset) + method +'/logs.pyd'): continue;
                try:
                    o = method_to_df(apath, setting, dataset, method)
                    for col in ignore_cols:
                        if col in o.columns:
                            o = o.drop([col], 1)
                    if methods is None:
                        methods = o.copy()
                    else:
                        methods = pd.concat([methods,o], axis=0, ignore_index=True, sort=False)
                except (pd.errors.ParserError, ValueError):
                    print('Could not parse', apath, setting, dataset, method)
        if methods is None:
            continue
        print('%d methods found' % len(methods))
        methods.fillna('', inplace=True)
        methods['conf_timestamp'] = pd.to_datetime(methods['conf_timestamp'])
        methods = methods[~methods['method'].isin(excluded_methods)]
        for c in methods.columns:
            if islist(methods[c]):
                methods[c] = methods[c].apply(str)
        
        methods_map_time = methods.conf_timestamp > '1980'
        # methods_map_time = methods.conf_timestamp > '2021-11-03 15:00:00'
        if methods_map_time.sum() < 1:
            continue

        # -------------------------- DISTINGUISH COLUMNS ----------------------------
        config_columns = [x for x in methods.columns if x.startswith('conf_')] +\
            ['seed', 'notes', 'csv_log', 'loss_log', 'examples_log', 'examples_full_log', 'tensorboard', 'savecheck', 'validation', 'dataset', 'ignore_other_metrics', 'balancoir', 'debug_mode','non_verbose',
            'distributed', 'start_from', 'intensive_savecheck', 'stopafter', 'loadcheck', 'autorelaunch', 'force_compat']
        config_columns = [x for x in config_columns if x in methods.columns]
        
        metric_columns = [x for x in methods.columns if any(map(lambda y: x.startswith(y), \
                            ['accuracy_', 'accmean_', 'forward_transfer', 'backward_transfer', 'forgetting']))]
        end = [x for x in methods.columns if x.startswith('accmean_')][-1]
       
        methods['aic'] = methods[[x for x in methods.columns if x.startswith('accmean_')]].mean(1)
        metric_columns.append('aic')
    
        parameter_columns = [x for x in methods.columns if x not in config_columns and x not in metric_columns]
        
        for c in metric_columns:
            methods[c] = pd.to_numeric(methods[c])
        distinct_cols = [x for x in t_distinct_cols if x in methods.columns]

        # -------------------------- GROUP BY PARAMETERS ----------------------------
        agg_dict = {end: ['count', aggf, 'std']}
        if len(metric_columns):
            agg_dict.update({x: ['count', aggf, 'std'] for x in metric_columns})
        agg_dict.update({'conf_jobnum': lambda x: str(list(x.values))})

        trez = methods[methods_map_time].groupby(parameter_columns).agg(agg_dict)
        trez.columns = ['%s-%s' % (a, b) for a,b in list(trez.columns)]
        trez = trez.reset_index().sort_values(by='%s-%s' % (end, 'aggf'), ascending=False)

        # -------------------------- PREPARE OUTPUT ----------------------------
        tmpout = trez[trez['%s-%s' %(end, 'count')] >= req_rep]

        tmpout['outmetric'] = pd.Series(['0'] * len(tmpout), index=tmpout.index).str.repeat((tmpout['%s-%s' %(end, 'aggf')] < 10).astype(int).values) + tmpout['%s-%s' %(end, 'aggf')].apply(lambda x: '%.2f' % x) + ' ± ' + tmpout['%s-%s' %(end, 'std')].apply(lambda x: '%.2f' % x)
        tmpout = tmpout.sort_values(by='outmetric', ascending=False)
        tmpout = tmpout.groupby(distinct_cols + ['conf_jobnum-<lambda>']).head(1)
        

        for m in [end] + (metric_columns if print_other_metrics else []):
            print('--- %s ---' % m)
            met1 = tmpout['%s-%s' %(m, 'aggf')].apply(lambda x: '%05.2f' % x)
            met2 = ' ± ' + tmpout['%s-%s' %(m, 'std')].apply(lambda x: '%.2f' % x)
            met3 = " [" + tmpout['%s-%s' %(m, 'count')].apply(lambda x: '%d' % x) #+ tmpout['conf_jobnum-<lambda>'].apply(lambda x: '] <a onclick="copyURI(event, \'%s\')">📋</a>' % str(x).replace('\'', '\\\''))
            met4 = ' (FG ' + tmpout['%s-%s' %('forgetting', 'aggf')].apply(lambda x: '%05.2f' % x) + ')'
            tmpout['tmpmetric'] = met1  + met2 + met3 + met4
            out = tmpout.groupby(distinct_cols)['tmpmetric'].max().unstack(-1).fillna('-')

            is_max = np.equal(out.values, out.max(axis=1).values[None].T).tolist()
        

            if len(out) > 0:
                #display(out.style.apply(highlight_max if m != 'forgetting' else highlight_min, axis=0))
                print(out)
        
    
            

    
        print('\n\n', '-' * 30, '\n\n')
