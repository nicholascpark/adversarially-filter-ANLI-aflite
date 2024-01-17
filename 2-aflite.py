"""
AFLite implements the AFLite (Algorithm 1) in https://arxiv.org/pdf/2002.04108.pdf
Run `pip install pyyaml scikit-learn` before running this script
"""
from numpy.core.fromnumeric import sort
import torch
import yaml
import numpy as np
from sklearn import svm

# We can train Pytorch model with GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using Device {DEVICE} For Pytorch computations")

def AFLite(phi, L, n, m, t, k, tau, output, use_numpy=False):
    """
    Input:
        phi: string path to load the data from.
        L: a trainable model.
        n: target dataset size.
        m: number of times the data is partitioned into two parts.
        t: size of the part used for training L.
        k: maximum number of rows to be deleted for each iteration.
            If an iteration chose less than k, we'll remove those rows and then terminate.
        tau: (0 to 1 ) predictability score to determine if a row should be removed. If a row has predictability score more or equal to tau, it'll be removed.
            1. the size of to be removed instances is less than k.
            2. size of the remaining data set is less than n.
        output: string path to write the filter to.

    Output
        Lx1 torch.Tensor, on each row, if true, keep the datapoint, if false, remove the datapoint.
    """
    print(f'Loading data from "{phi}" and output to "{output}"')
    print(f'Model = {L}, n={n}, m={m}, t={t}, k={k}, tau={tau}')
    # load dataset represented as a pytorch dataset
    phi = torch.load(phi)   
    print(f'Loaded Phi={phi.shape}')
    assert n < phi.shape[0], f'Loaded dataset size must be greater than target dataset size, got n {n} >= X {phi.shape[0]}'
    assert t < phi.shape[0], f'Loaded dataset size must be greater than training dataset size, got t {t} >= X {phi.shape[0]}'
    assert n == 0 or t < n, f'n must be 0 or training dataset size must be smaller than target dataset size, got t {t} >= n {n}'
    assert n == 0 or k <= n, f'n must be 0 or slice size must be smaller or equal to target dataset size, got k {k} > n {n}'
    # Augment the input matrix with original indices.
    position = torch.arange(0, phi.shape[0], dtype=torch.int).unsqueeze(1)
    S = torch.cat((phi, position), 1)
    # S contains [features..., label, original row index]

    itr_count = 0
    while S.shape[0] > n:
        print(f'Iteration {itr_count}, size {S.shape[0]}')
        itr_count += 1

        # Any row not getting selected will have a predictability score of 0.
        E = cross_validation(L, S, t, m, use_numpy, monte_carlo=True)
        assert E.shape[0] == S.shape[0]
        before = S.shape[0]
        sortedE, indice = torch.sort(E)  # we rank the instances according to their predictability score
        # small to large
        print("Max probability:", sortedE[-1], "Average probability:", torch.sum(sortedE) / sortedE.shape[0])
        mask = (sortedE < tau)  # we remove the top-k instances whose score is not less than the early-stoppipng threshold tau.
        mask[:-k] = True  # we keep the ones up to the last k instances ( remove larger ones )
        S = S[indice[mask]]
        if before - S.shape[0] < k:
            print(f'Iteration {itr_count}, size reduction {before - S.shape[0]} is smaller than k {k}, break')
            break

    out = torch.zeros((phi.shape[0],), dtype=torch.bool)
    out[S[:,-1].long()] = True  # Only remaining stuff in S will be used for the next step.
    torch.save(out, output)
    print(f'Iteration {itr_count}, final size is {S.shape[0]}')
    pass

def cross_validation(L, S: torch.Tensor, t: int, m: int, use_numpy=False, monte_carlo=True):
    # split the training set randomly into two parts
    # S must have: last index as the original index, -2 index as the label
    if use_numpy:
        S = S.numpy()
    else:
        S = S.to(DEVICE)

    # (number of time it gets right, number of time it gets selected)
    E = torch.zeros((S.shape[0], 2), dtype=torch.float)
    if not use_numpy:
        E = E.to(DEVICE)

    if monte_carlo:
        for _ in range(m):
            rand_indices = torch.randperm(S.shape[0])
            indices = torch.arange(0, S.shape[0], dtype=torch.long)[rand_indices]
            S = S[rand_indices]
            if use_numpy:
                # randomly split S into 2
                Tj, S_Tj = S[:S.shape[0] - t], S[S.shape[0] - t:]
                selected_for_prediction = indices[:S.shape[0] - t]
                assert Tj.shape[0] == S.shape[0] - t
                assert S_Tj.shape[0] == t
                # prepare validation and training input and labels
                X_S_Tj, Y_S_Tj = S_Tj[:,:-2], S_Tj[:,-2].astype(np.int8)
                X_Tj, Y_Tj = Tj[:,:-2], Tj[:,-2].astype(np.int8)
                # L computes the correct matrix containing boolean of correct predictions in validation set.
                correct = L(X_S_Tj, Y_S_Tj, X_Tj, Y_Tj)
                assert correct.shape[0] == Tj.shape[0]
                # Correct prediction is 1 while incorrect is 0, we increase them.
                E[selected_for_prediction, 0] += correct.astype(np.int8)
                E[selected_for_prediction, 1] += 1
            else:
                # validation set, training set
                Tj, S_Tj = torch.split(S, (S.shape[0] - t, t))
                # Tj is validation set, S_Tj is training set
                selected_for_prediction = indices[:S.shape[0] - t]
                X_S_Tj, Y_S_Tj = S_Tj[:,:-2], S_Tj[:,-2].long()
                X_Tj, Y_Tj = Tj[:,:-2], Tj[:,-2].long()
                correct = L(X_S_Tj, Y_S_Tj, X_Tj, Y_Tj)
                E[selected_for_prediction, 0] += correct.int()
                E[selected_for_prediction, 1] += 1
    else:
        raise Exception('Not implemented yet')

    selected = E[:,1] != 0  # avoid NAN, any unselected data point will receive 0.
    E[selected, 0] = E[selected, 0] / E[selected, 1]
    return E[:,0]

def _svm(X, y, X_validation, y_validation):
    clf = svm.SVC()  # default kernel = RBF
    clf.fit(X, y)
    pred = clf.predict(X_validation)
    return (pred == y_validation)

def _linearSGD(X, y, X_validation, y_validation):
    unique = torch.unique(y)
    classes = unique.shape[0]
    model = torch.nn.Linear(X.shape[1], classes).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()  # this contains a LogSoftmax step
    epoch = 10
    for _ in range(epoch):
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        outputs = model(X_validation)
        _, predicted = torch.max(outputs.data, 1)
        return (predicted == y_validation)


models = {
    'SVM': _svm,
    'LinearSGD': _linearSGD,
}

models_numpy = {
    'SVM': True,
    'LinearSGD': False,
}

def run():
    with open('config.yaml', 'r') as y:
        config = {
            'phi': 'Location to load the embeddings',
            'L': 'SVM',
            'm': 64,
            't': 40000,
            'tau': 0.75,
            'k': 10000,
            'n': 640000, # relavent to size of the input
            'output': 'Location to output the filter',
        }

        for k, v in yaml.safe_load(y)['2-aflite'].items():
            config[k] = v

        model = models[config['L']]
        AFLite(config['phi'], model, config['n'], config['m'], config['t'], config['k'],
            config['tau'], config['output'], use_numpy=models_numpy[config['L']])

if __name__ == '__main__':
    run()