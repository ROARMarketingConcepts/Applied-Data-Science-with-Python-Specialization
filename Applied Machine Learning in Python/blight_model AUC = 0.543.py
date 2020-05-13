def blight_model():
    
    import sys
    import warnings

    if not sys.warnoptions:                       # ignore minor warnings while code is running
        warnings.simplefilter("ignore")
    
    import pandas as pd
    import numpy as np
    import datetime as dt
    # import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import confusion_matrix,roc_curve, auc
    from adspy_shared_utilities import plot_class_regions_for_classifier
    
    train = pd.read_csv('train.csv',engine='python')
    
    # Select the columns in the train dataset that we will keep
    
    cols_to_keep = ['ticket_id','violation_code','judgment_amount','compliance_detail','compliance']
    train = train[cols_to_keep]
    
    # Set 'ticket_id' as the row index
    
    train.set_index('ticket_id',inplace=True)
    
    # Delete rows in 'train' where no violation has occurred

    train = train[(train['compliance_detail'] != 'not responsible by disposition') & 
                  (train['compliance_detail'] != 'not responsible by pending judgment disposition')]
    
    # Clean up the target 'compliance' column
    
    train['compliance'] = 1 
    non_compliant = train['compliance_detail'].str.contains('non-compliant',regex=False)
    train['compliance'][non_compliant] = 0
    
    # Now we can get rid of the 'compliance_detail' column
    
    train.drop('compliance_detail', axis=1, inplace=True)
    
    # Recode the 'violation_code' column
    
    train['violation_recode'] = 'other'
    code9_1 = train['violation_code'].str.startswith('9-1')
    train['violation_recode'][code9_1] = '9-1'
    code22_2 = train['violation_code'].str.startswith('22-2')
    train['violation_recode'][code22_2] = '22-2'
    code61 = train['violation_code'].str.startswith('61')
    train['violation_recode'][code61] = '61'
    code194 = train['violation_code'].str.startswith('194')
    train['violation_recode'][code194] = '194'
    
    # Convert 'violation_recode' to a coded categorical variable
    train['violation_recode'] = train['violation_recode'].astype('category')
    train['violation_recode'] = train['violation_recode'].cat.codes
    
    # Now we can get rid of the 'violation_code' column
    train.drop('violation_code', axis=1, inplace=True)
    
    # Create the X and y dataframes
    
    X = train.drop(['compliance'],axis=1)
    y = train['compliance']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # We must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)
    
    # Set up the Gradient Boosting Classifier
    
    clf = GradientBoostingClassifier(random_state = 0)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    confusion = confusion_matrix(y_test, y_pred)
    
    y_score = clf.fit(X_train_scaled, y_train).decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    '''print('Accuracy of GBDT classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
    print('Accuracy of GBDT classifier on test set: {:.2f}\n'.format(clf.score(X_test_scaled, y_test)))
    print(confusion)
    
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr, tpr, lw=3, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    # plt.axes().set_aspect('equal')
    plt.show()'''
    
    # Now, let's process the test dataset
    
    test = pd.read_csv('test.csv',engine='python')
    cols_to_keep = ['ticket_id','violation_code','judgment_amount']
    test = test[cols_to_keep]
    ticket_id = test['ticket_id']
    test.set_index('ticket_id',inplace=True)

    # Recode the 'violation_code' column
    
    test['violation_recode'] = 'other'
    code9_1 = test['violation_code'].str.startswith('9-1')
    test['violation_recode'][code9_1] = '9-1'
    code22_2 = test['violation_code'].str.startswith('22-2')
    test['violation_recode'][code22_2] = '22-2'
    code61 = test['violation_code'].str.startswith('61')
    test['violation_recode'][code61] = '61'
    code194 = test['violation_code'].str.startswith('194')
    test['violation_recode'][code194] = '194'
    
    # Convert 'violation_recode' to a coded categorical variable
   
    test['violation_recode'] = test['violation_recode'].astype('category')
    test['violation_recode'] = test['violation_recode'].cat.codes
    
    # Now we can get rid of the 'violation_code' column
    test.drop('violation_code', axis=1, inplace=True)
    
    test_scaled = scaler.fit_transform(test)
    
    # Now get the probabilities of a '1' outcome using the 'proba' function
    
    y_probs = clf.fit(X_train_scaled, y_train).predict_proba(test_scaled)
    
    result = pd.Series(y_probs[:,1], index = ticket_id)   # create a pandas Series with the ticket_id as index
    
    return result

blight_model()
