from .cnn import CNN
from .resnet import ResNet18
from .resnet_legacy import resnet18
from .lstm import LSTMForecast


def get_model(args):
    ## User-defined model
    model = None
    if args.model == "CNN":
        model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    elif args.model == "resnet18":
        model = ResNet18()
    elif args.model == "resnet18-legacy":
        model = resnet18(args.num_channel, args.num_classes, args.pretrained)
    elif args.model == "LSTM":
        model = LSTMForecast(
            n_features=args.n_features,
            n_lookback=args.n_lookback,
            n_lstm_layers=args.n_lstm_layers,
            n_hidden_size=args.n_hidden_size,
        )
    else:
        raise NotImplementedError
    return model


def validate_parameter_names(model, list_of_params):
    # inputs model and list with name of parameters
    # returns IS_VALID, IS_EMPTY

    if list_of_params == []:
        IS_VALID = True
        IS_EMPTY = True
        return IS_VALID, IS_EMPTY

    model_keys = [key for key, _ in model.named_parameters()]

    for p in list_of_params:
        if p not in model_keys:
            IS_VALID = False
            IS_EMPTY = False
            return IS_VALID, IS_EMPTY

    IS_VALID = True
    IS_EMPTY = False
    return IS_VALID, IS_EMPTY
