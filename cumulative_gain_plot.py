# Function for creating Cumulative Gain plot for a binary classifier
# Return 3 lists:
# accumulative gain
# sample population cumulative
# descending predicted probability

import matplotlib.pyplot as plt
from sklearn.metrics import auc

def cumulative_gain_plot(model, X, y,label=None):

    # extract predicted class probability to be positive
    proba_pos = model.predict_proba(X)[:,1]

    # create np.array of actual positive class label
    actual_pos = np.array(list(y))
    
    # reshape numpy arrays to (:, 1) and join predicted and actual class label to generate 2D array
    joint = np.concatenate([proba_pos.reshape(proba_pos.shape[0],1), actual_pos.reshape(actual_pos.shape[0],1)], 
                           axis=1)
    
    # sort 2D array in ascending order by the column of predicted class probability
    joint_sort=np.sort(joint.view('i8,i8'), order=['f0'], axis=0).view(np.float)
    
    # calculate the number of total sample population and actual positive class
    total_num = len(joint_sort)
    total_pos = sum(joint_sort[:,1]==1)

    # Descending order of actual class label in the order from high to low probability
    proba_pos_desc = joint_sort[:,1][::-1]
    
    # Descending order of predicted probability in the order from high to low
    # will be used for identifying customarized probability threshold in y_predict_threshold function
    proba_pred_desc = joint_sort[:,0][::-1] 

    # create positive cumulative response%
    pos_cumu = 0
    pos_cumu_list = [0]
    for i in range(len(joint_sort)):
        pos_cumu = pos_cumu + proba_pos_desc[i]/total_pos
        pos_cumu_list.append(pos_cumu)
        
    # create cumulative population %
    pop_cumu = 0
    pop_cumu_list = [0]
    for i in range(len(joint_sort)):
        pop_cumu = pop_cumu + i/total_num
        pop_cumu_list.append(i/total_num)

    # create cumulative gain plot
    plt.plot(pop_cumu_list, pos_cumu_list, label=label)
    plt.plot([0,1],[0,1], 'k--') # reference line for random model
    plt.axis([0,1,0,1])
    plt.xlabel('Population%')
    plt.ylabel('Response%')

    # calculate area under curve
    auc_score = auc(pop_cumu_list, pos_cumu_list, reorder=False)
    print('AUC score of %s is %.4f.' % (label, auc_score))

    return pos_cumu_list, pop_cumu_list, proba_pred_desc
