import argparse


from model import PoligrasRunner


def main():

    parser = argparse.ArgumentParser(description="Run Poligras.")
    parser.add_argument("--dataset", nargs="?", default="in-2004", help="Dataset name")
    parser.add_argument("--counts", type=int, default=100, help="Number of graph summarization iterations.")
    parser.add_argument("--group_size", type=int, default=200, help="Size of each divided group in the group partitioning stage.")
    parser.add_argument("--hidden_size1", type=int, default=64, help="1st hidden layer size of MLP in the policy function.")
    parser.add_argument("--hidden_size2", type=int, default=32, help="Output layer size of MLP in the policy function.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the back-propagation.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability in the policy function for the node pair selection probability matrix computation.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam optimizer weight decay.")
    parser.add_argument("--bad_counter", type=int, default=0, help="Value of bad counts to tolerate for each training iteration.")
    args = parser.parse_args()


    executer = PoligrasRunner(args)
    executer.fit()
    executer.encode()
    

if __name__ == "__main__":
    main()
