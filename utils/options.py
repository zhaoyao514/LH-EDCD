import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # classic FL settings
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.8, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B, default: 128")
    parser.add_argument('--local_test_bs', type=int, default=128, help="test batch size, default: 128")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # Model and Datasets
    # model arguments, support model: "cnn", "mlp"
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    # support dataset: "kdd99" and "CIC-IDS2017"
    parser.add_argument('--dataset', type=str, default='kdd99', help="name of dataset: kdd99 or CIC-IDS2017")
    # total used dataset size for all nodes
    parser.add_argument('--dataset_size', type=int, default=4000, help="total used dataset size for all nodes")

    # env settings
    parser.add_argument('--fl_listen_port', type=str, default='8888', help="federated learning listen port")
    parser.add_argument('--numpy_rdm_seed', type=int, default=8888, help="the random seed of numpy for unified dataset"
                                                                         " distribution, -1 for no seed")
    # test global model accuracy on all nodes or on a central node
    parser.add_argument('--dis_acc_test', action='store_true', help='central or distributed accuracy test')
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--log_level', type=str, default='DEBUG', help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')
    # ip address that is used to test local IP
    parser.add_argument('--test_ip_addr', type=str, default="10.150.187.13", help="ip address used to test local IP")
    # sleep for several seconds before start train
    parser.add_argument('--start_sleep', type=int, default=5, help="sleep for seconds before start train")
    # sleep for several seconds before exit python
    parser.add_argument('--exit_sleep', type=int, default=0, help="sleep for seconds before exit python")

    args = parser.parse_args()
    return args
