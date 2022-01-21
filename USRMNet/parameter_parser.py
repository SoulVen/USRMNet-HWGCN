""" Parsing the parameters """
import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Running uRLLCUnfolding.")

    parser.add_argument("--K", type=int, default=4,
                        help="Total number of users. Default is 4.")

    parser.add_argument("--Nt", type=int, default=32,
                        help="Number of transmitting anttennas. Default is 32.")

    parser.add_argument("--du", type=int, default=120,
                        help="The minimum distance between the BS and UE.")

    parser.add_argument("--dc", type=int, default=140,
                        help="The radius of cell.")

    parser.add_argument("--Vartheta", type=float, default=0.2666,
                        help="vartheta. Default is 0.377.")

    parser.add_argument("--num-H", type=int, default=10000,
                        help="Number of training samples. Default is 5000.")

    parser.add_argument("--num-test", type=int, default=20,
                        help="Number of testing samples. Default is 200.")

    parser.add_argument("--outer-itr", type=int, default=5,
                        help="Number of fixed outer module layers. Default is 1.")

    parser.add_argument("--max-itr", type=int, default=1,
                        help="Number of fixed inner module layers. Default is 1.")

    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training iterations. Default is 1000.")

    parser.add_argument("--batchsize", type=int, default=20,
                        help="Batchsize. Default is 20.")

    parser.add_argument("--sigmma", type=int, default=1,
                        help="sigmma value. Default is 1.")

    parser.add_argument("--D", type=int, default=256,
                        help="length of transmitting data. Default is 256.")

    parser.add_argument("--n", type=int, default=256,
                        help="finite blocklength. Default is 128.")

    parser.add_argument("--Pmax", type=float, default=10**(15/10),
                        help="SNR value. limitation of maximum total power is 20.")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate of adam optimizer. Default is 0.003.")

    parser.add_argument("--lr-coe", type=float, default=0.0001,
                        help = "step size of Lagrangian Multipliers. Default is 0.0001.")
    return parser.parse_args()