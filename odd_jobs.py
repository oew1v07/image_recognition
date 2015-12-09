from im_recog import *

test_list_of_jpgs = list_of_jpegs
histograms = np.load(join('/Users/robin/COMP6223/cw3/', 'run3_histograms.npy'))
targets = np.load(join('/Users/robin/COMP6223/cw3/', 'run3_targets.npy'))

ovr, acc_tr, acc_tst, full_tr_acc = one_vs_all(histograms, targets,
                                               test_size=test_size,
                                               run_num=run_num,
                                               svm_type='non-linear')

acc_tr = np.array(acc_tr)
acc_tst = np.array(acc_tst)

# Saving accuracies
np.save('run3_acc_tr', acc_tr)
np.save('run3_acc_tst', acc_tst)
np.save('run3_full_tr_acc', full_tr_acc)

np.load('run3_test_histograms.npy')

predicted_class = ovr.predict(test_histograms)

write_output(test_list_of_jpgs, predicted_class, run_no=3)


################################################################################

# Do accuracy of ovr with different ks for the nearest neighbour part. RUN3
ks = [1,3,5]
for k in ks:

    [histograms, targets,
     neigh] = get_training_data_for_histogram(la_list_of_centres,
                                              la_list_of_words,
                                              lla_daisy_of_each_image,
                                              order_of_classes,
                                              run_no=3, k=k)

    joblib.dump(neigh, 'run3_neigh_k_{}.pkl'.format(k))

    ovr, acc_tr, acc_tst = one_vs_all(histograms, targets,
                                      test_size=test_size, run_num=run_num,
                                      svm_type='non-linear')


    joblib.dump(ovr, 'run3_ovr_k_{}.pkl'.format(k))


################################################################################

# Do accuracy of ovr with different samples for clustering
