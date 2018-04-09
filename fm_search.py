def build_fm_model(X, y, X_test, y_test, n_iter=2, init_stdev=0.1, rank=8, l2_reg_w=0.1, l2_reg_V=0.1):
    """X: training features, must be scipy.sparse.csc_matrix
    y: training labels, must be -1 or 1.
    X_test: test data 
    y_test: test labels
    n_iter: number of iterations desired for model fitting
    init_stdev: std for weight initalisation
    rank: number of latent factors for model
    l2_reg_w: regularisation param for linear weights
    le_reg_V: regularisation param for pairwise weights"""

    #start timer
    import time
    start = time.time()
    import scipy 
    output = {}
    from fastFM import als

    print('//////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('Fitting FM model with {} iterations, {} init_stdev, {} rank, {} l2_reg_w, {} l2_reg_V'.format(n_iter, init_stdev, rank, l2_reg_w, l2_reg_V))
    print('...')
    model = als.FMClassification(n_iter, init_stdev, rank, l2_reg_w, l2_reg_V)
    model.fit(X,y)
    
    #predictions for training and test set
    train_pred = model.predict(X)
    train_pred_proba = model.predict_proba(X)
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)
    time_taken = time.time()-start
    output.update({'Time':time_taken, 'train_acc': accuracy_score(y, train_pred), 'test_acc': accuracy_score(y_test, test_pred),
                    'train_ll': log_loss(y, train_pred_proba), 'test_ll':log_loss(y_test, test_pred_proba),
                   'train_auc': roc_auc_score(y, train_pred_proba), 'test_auc':roc_auc_score(y_test, test_pred_proba),
                  'conf_train': confusion_matrix(y, train_pred), 'conf_test': confusion_matrix(y_test, test_pred)})
    print('Training accuracy: {}'.format(output['train_acc']))
    print('Test accuracy: {}'.format(output['test_acc']))
    print('Training AUC: {}'.format(output['train_auc']))
    print('Test AUC: {}'.format(output['test_auc']))
    print('Training log loss: {}'.format(output['train_ll']))
    print('Test log loss: {}'.format(output['test_ll']))
    print('Time taken: ', output['Time'])
    print('')
    print('...')
    print('/////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print(' ')
    return(model, output)    

def hyper_search(X, y, X_test, y_test, params):
    """X: training features, must be scipy.sparse.csc_matrix
    y: training labels, must be -1 or 1.
    X_test: test data 
    y_test: test labels
    params: a list of parameter dictionariest, can be made from a sklearn.model_selection parameter grid list
    """  
    grid_search = []
    models = []
    for params in param_list:
        model, out = build_fm_model(X, y, X_test, y_test, **params)
        grid_search.append({**params, **out})
        models.append(model)
    return(models, grid_search)