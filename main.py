from scripts.train import train
from scripts.train import test
import argparse

parser = argparse.ArgumentParser(description='Run the feedback network.')
parser.add_argument('--checkpoint', type=str, 
                    help='checkpoint to load the model from', default=None)
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use the GPU for training/testing')
parser.add_argument('--train', action='store_true',
                    help='train the network')
parser.add_argument('--test', action='store_true', default=False,
                    help='test the network')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='dataset to use. possibilities: CIFAR10, CIFAR100')
parser.add_argument('--network', type=str, default='feedback48',
                    help='network to use. possibilities: feedbacknet, feedback48')
parser.add_argument('--classes', type=int, default=10,
                    help="number of output classes")

if __name__ == '__main__':
  args = parser.parse_args()
  print(args.__dict__)
  if args.train:
    net = train(network=args.network, checkpoint=args.checkpoint, cuda=args.cuda, 
                epochs=args.epochs, dataset=args.dataset, n_classes=args.classes)
    if args.test:
      test(cuda=args.cuda, test_network=net)
  elif args.test:
    test(network=args.network, checkpoint=args.checkpoint, cuda=args.cuda, 
         dataset=args.dataset, n_classes=args.classes)
