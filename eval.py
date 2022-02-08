import argparse
import torch
from data import JamendoLyricsDataset
from model import AcousticModel
import utils, test

def main(args):

    if args.model == "baseline":
        n_class = 41
    elif args.model == "MTL":
        n_class = (41, 47)
    else:
        ValueError("Invalid model type.")

    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 3,
        "rnn_dim": args.rnn_dim,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    model = AcousticModel(
        hparams['n_cnn_layers'], hparams['rnn_dim'], hparams['n_class'], \
        hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    if 'cuda' in device:
        print("move model to gpu")
        model = utils.DataParallel(model)
        model.cuda()

    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    print("Loading full model from checkpoint " + str(args.load_model))

    state = utils.load_model(model, args.load_model, args.cuda)

    test_data = JamendoLyricsDataset(args.sr, args.hdf_dir, args.dataset, args.jamendo_dir, args.sepa_dir, unit=args.unit)

    results = test.predict_align(args, model, test_data, device, args.model)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--jamendo_dir', type=str, required=True,
                        help='Dataset path')
    parser.add_argument('--sepa_dir', type=str, required=True,
                        help='Where all the separated vocals of Jamendo are stored.')
    parser.add_argument('--dataset', type=str, default="jamendo",
                        help='Dataset name')
    parser.add_argument('--hdf_dir', type=str, default="./hdf/",
                        help='Dataset path')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Prediction path')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--model', type=str, default="baseline",
                        help='"baseline" or "MTL"')
    parser.add_argument('--sr', type=int, default=22050,
                        help="Sampling rate")
    parser.add_argument('--rnn_dim', type=int, default=256,
                        help="RNN dimension")
    parser.add_argument('--unit', type=str, default="phone",
                        help="Alignment unit: char or phone; Should match the model type.")

    args = parser.parse_args()

    main(args)