import os, argparse
import torch
from data import JamendoLyricsDataset
from model import AcousticModel, BoundaryDetection
import utils, test

def main(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    # acoustic model
    if args.model == "baseline":
        n_class = 41
    elif args.model == "MTL":
        n_class = (41, 47)
    else:
        ValueError("Invalid model type.")

    ac_hparams = {
        "n_cnn_layers": 1,
        "rnn_dim": args.rnn_dim,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    ac_model = AcousticModel(
        ac_hparams['n_cnn_layers'], ac_hparams['rnn_dim'], ac_hparams['n_class'], \
        ac_hparams['n_feats'], ac_hparams['stride'], ac_hparams['dropout']
    ).to(device)

    # boundary model: fixed
    bdr_hparams = {
        "n_cnn_layers": 1,
        "rnn_dim": 32, # a smaller rnn dim than acoustic model
        "n_class": 1, # binary classification
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1,
    }

    bdr_model = BoundaryDetection(
        bdr_hparams['n_cnn_layers'], bdr_hparams['rnn_dim'], bdr_hparams['n_class'],
        bdr_hparams['n_feats'], bdr_hparams['stride'], bdr_hparams['dropout']
    ).to(device)

    if 'cuda' in device:
        print("move model to gpu")
        ac_model = utils.DataParallel(ac_model)
        ac_model.cuda()
        bdr_model = utils.DataParallel(bdr_model)
        bdr_model.cuda()

    print('parameter count (acoustic model): ', str(sum(p.numel() for p in ac_model.parameters())))
    print('parameter count (boundary model): ', str(sum(p.numel() for p in bdr_model.parameters())))

    print("Loading full model from checkpoint " + str(args.ac_model))
    print("Loading full model from checkpoint " + str(args.bdr_model))

    ac_state = utils.load_model(ac_model, args.ac_model, args.cuda)
    bdr_state = utils.load_model(bdr_model, args.bdr_model, args.cuda)

    test_data = JamendoLyricsDataset(args.sr, args.hdf_dir, args.dataset, args.jamendo_dir, args.sepa_dir, unit=args.unit)

    # predict with boundary detection
    results = test.predict_w_bdr(args, ac_model, bdr_model, test_data, device,
                                 args.alpha, args.model)

if __name__ == '__main__':
    ## EVALUATE PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=24,
                        help='Number of feature channels per layer')
    parser.add_argument('--jamendo_dir', type=str, required=True,
                        help='Dataset path')
    parser.add_argument('--sepa_dir', type=str, required=True,
                        help='Where all the separated vocals of Jamendo are stored.')
    parser.add_argument('--dataset', type=str, default="jamendo",
                        help='Dataset name')
    parser.add_argument('--hdf_dir', type=str, default="./hdf/",
                        help='Dataset path')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='prediction path')
    parser.add_argument('--ac_model', type=str, required=True,
                        help='Reload a previously trained acoustic model')
    parser.add_argument('--bdr_model', type=str, required=True,
                        help='Reload a previously trained boundary detection model')
    parser.add_argument('--model', type=str, default="baseline",
                        help='"baseline" or "MTL"')
    parser.add_argument('--sr', type=int, default=22050,
                        help="Sampling rate")
    parser.add_argument('--rnn_dim', type=int, default=256,
                        help="RNN dimension")
    parser.add_argument('--unit', type=str, default="phone",
                        help="Alignment unit: char or phone; Should match the model type.")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="weight svd score")

    args = parser.parse_args()

    main(args)

    main(args)