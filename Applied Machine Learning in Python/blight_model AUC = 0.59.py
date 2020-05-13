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
    
    cols_to_keep = ['ticket_id','zip_code','violation_code','judgment_amount','compliance_detail','compliance']
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
    
    # Recode the 'zip_code' column 
    
    train['zip_code'] = train['zip_code'].astype('str')
    
    # Find all rows where zip code does not start with '48'
    
    def recode_odd_ball_zips(df):
    
        good_zips = ['48227','48221','48235','48228','48219','48238','48224','48205',
                 '48204','48234','48206','48213','48223','48209','48203','48075',
                 '48210','48207','48202','48076','48214','48226','48212','48037',
                 '48034','48215','48237','48208','48126','48201','48126']
    
        odd_ball_zips = df.loc[~df['zip_code'].str.startswith(('48'))]  
    
        increment = int(len(odd_ball_zips)/len(good_zips))+1
    
        for i,zip_code in enumerate(good_zips):                             # populate 'rows['zip_code]' with zip codes from 'good_zips'
            odd_ball_zips['zip_code'][increment*i:increment*(i+1)] = zip_code
        
        df.update(odd_ball_zips)                                             # update the dataset with the recoded zip codes.
        
        return df
    
    train = recode_odd_ball_zips(train)

    # Convert columns of strings to coded categorical variables
    
    def categorical(series):
        series = series.astype('category')
        series = series.cat.codes
        return series
    
    train['zip_code'] = categorical(train['zip_code'])

    # Recode the 'violation_code' column and convert to coded categorical variable
    
    def recode_violation_code_column(df):
        
        df['violation_recode'] = 'other'
        code9_1 = df['violation_code'].str.startswith('9-1')
        df['violation_recode'][code9_1] = '9-1'
        code22_2 = df['violation_code'].str.startswith('22-2')
        df['violation_recode'][code22_2] = '22-2'
        code61 = df['violation_code'].str.startswith('61')
        df['violation_recode'][code61] = '61'
        code194 = df['violation_code'].str.startswith('194')
        df['violation_recode'][code194] = '194'
        df['violation_recode'] = df['violation_recode'].astype('category')
        df['violation_recode'] = df['violation_recode'].cat.codes
        df.drop('violation_code', axis=1, inplace=True)
        
        return df
    
    train = recode_violation_code_column(train)
    
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
    cols_to_keep = ['ticket_id','zip_code','violation_code','judgment_amount']
    test = test[cols_to_keep]
    
    ticket_id = test['ticket_id']                # save the 'ticket_id' column in test, we're going to need it later
    test.set_index('ticket_id',inplace=True)

    # Recode the 'zip_code' column
    
    test['zip_code'] = test['zip_code'].astype('str')
    test = recode_odd_ball_zips(test)
    test['zip_code'] = categorical(test['zip_code'])
    
    # Recode the 'violation_code' column
    
    test = recode_violation_code_column(test)
    
    # Min-Max scale the numerical variables
    
    test_scaled = scaler.fit_transform(test)
    
    # Now get the probabilities of a '1' outcome using the 'proba' function
    
    y_probs = clf.fit(X_train_scaled, y_train).predict_proba(test_scaled)
    
    compliance = pd.Series(y_probs[:,1], index = ticket_id)   # create a pandas Series with the ticket_id as index'''
    
    return compliance

blight_model()