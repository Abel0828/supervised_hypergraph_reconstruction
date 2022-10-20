from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
eps = 1e-4


def get_model(args, dataloader):
    if args.model == 'rf':
        model1 = RandomForestClassifier(random_state=args.seed, n_jobs=-1, verbose=0)
    elif args.model == 'mlp':
        model1 = MLPClassifier(random_state=args.seed, verbose=False, max_iter=args.epochs, alpha=1e-4)
    elif args.model == 'lr':
        model1 = LogisticRegression(random_state=args.seed, n_jobs=-1)
    else:
        raise Exception
    return model1