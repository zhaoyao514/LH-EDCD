import numpy as np


def get_indices(labels, user_labels, user_label_sizes):
    indices = []
    for i in range(len(user_labels)):
        label_samples = np.where(labels[1, :] == user_labels[i])
        label_indices = labels[0, label_samples]
        if user_label_sizes[i] < len(label_indices[0]):
            selected_indices = list(np.random.choice(label_indices[0], user_label_sizes[i], replace=False))
        else:
            selected_indices = list(label_indices[0])
        # print("selected user label: {} \n selected user label number: {} \n"
        #       .format(user_labels[i], len(selected_indices)))
        indices += selected_indices
    return indices


def get_user_indices(dataset_name, dataset_train, dataset_test, dataset_size, num_users):
    train_users = {}
    test_users = {}

    train_idxs = np.arange(len(dataset_train))
    train_labels = dataset_train.targets
    train_labels = np.vstack((train_idxs, train_labels))

    test_idxs = np.arange(len(dataset_test))
    test_labels = dataset_test.targets
    test_labels = np.vstack((test_idxs, test_labels))

    if dataset_name == 'kdd99':
        data_classes = 2
    else:
        data_classes = 0
    labels = list(range(data_classes))

    for i in range(num_users):
        # choose all labels in a random order
        user_labels = np.random.choice(labels, size=data_classes, replace=False)
        # The dataset size used for training. If it is variant, the number is calculated according to the user id
        train_sample_size = round(dataset_size / num_users)
        # test size : train size = 2 : 10
        test_sample_size = round(train_sample_size / 5)

        # calculate train sample sizes for all classes
        user_train_label_size_list = [round(train_sample_size / data_classes)] * data_classes
        user_test_label_size_list = [round(test_sample_size / data_classes)] * data_classes

        train_indices = get_indices(train_labels, user_labels, user_train_label_size_list)
        test_indices = get_indices(test_labels, user_labels, user_test_label_size_list)

        train_users[i] = train_indices
        test_users[i] = test_indices
    return train_users, test_users
