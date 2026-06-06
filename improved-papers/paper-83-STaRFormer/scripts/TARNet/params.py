import pickle

if __name__ == '__main__':
    with open('./scripts/TARNet/tarnet/hyperparameters.pkl', 'rb') as f:
        hyparmas = pickle.load(f)

    print(hyparmas)
    print(hyparmas['AE'])