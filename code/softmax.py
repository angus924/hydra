# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://arxiv.org/abs/2203.13652

import copy
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim

from hydra import Hydra

def train(path, num_classes, training_size, **kwargs):

    # -- init ------------------------------------------------------------------

    args = \
    {
        "validation_proportion" : 0.1,
        "validation_min"        : 1_024,
        "chunk_size"            : 2 ** 12,
        "chunk_size_sgd"        : 2 ** 12,
        "minibatch_size"        : 256,
        "max_epochs"            : 200,
        "patience_lr"           : 10,
        "patience"              : 20,
        "threshold"             : 1e-4,
        "k"                     : 8,
        "g"                     : 64,
        "seed"                  : None,
        "validate"              : True
    }
    args = {**args, **kwargs}

    if args["seed"] is not None:
        torch.manual_seed(args["seed"])

    random_state = torch.random.get_rng_state()

    data = np.load(path, mmap_mode = "r")

    max_size = data.shape[0]

    _validation_size = max(np.int32(max_size * args["validation_proportion"]), args["validation_min"])

    # -- validation data -------------------------------------------------------

    indices = torch.randperm(max_size)
    validation_indices, training_indices = indices[:_validation_size], indices[_validation_size:]

    validation_data = torch.tensor(data[validation_indices])
    X_validation, Y_validation = validation_data[:, :-1].float().unsqueeze(1), validation_data[:, -1].long()

    transform = Hydra(X_validation.shape[-1], k = args["k"], g = args["g"], seed = None)

    X_validation_transform = transform.batch(X_validation).clamp(0).sqrt()
    validation_mask = X_validation_transform != 0

    # -- init (cont) -----------------------------------------------------------

    exponent = np.log2((X_validation.shape[-1] - 1) / (9 - 1))
    num_dilations = int(exponent) + 1
    _num_features = num_dilations * 2 * 512

    def init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight.data, 0)
            nn.init.constant_(layer.bias.data, 0)

    # -- cache -----------------------------------------------------------------

    cache_Y = torch.zeros(training_size, dtype = torch.long)
    cache_X = torch.zeros((training_size, _num_features))

    cache_map = torch.zeros(max_size).long()

    torch.random.set_rng_state(random_state)

    chunks = training_indices[torch.randperm(training_size)].split(args["chunk_size"])
    sequences = torch.arange(training_size).split(args["chunk_size"])

    f_mean = 0
    f_std = 0

    est_size = 0

    for chunk_index, chunk in enumerate(chunks):

        chunk_size = len(chunk)

        training_data = np.array(data[chunk])
        X_training, Y_training = torch.FloatTensor(training_data[:, :-1]).unsqueeze(1), torch.LongTensor(training_data[:, -1])

        X_training_transform = transform.batch(X_training).clamp(0).sqrt()

        s = (X_training_transform == 0).float().mean(0) ** 4 + 1e-8

        cache_map.scatter_(-1, chunk, sequences[chunk_index])

        cache_indices = cache_map.gather(-1, chunk)

        cache_X[cache_indices] = X_training_transform
        cache_Y[cache_indices] = Y_training

        _f_mean = X_training_transform.mean(0)
        _f_std = X_training_transform.std(0) + s

        f_mean = ((f_mean * est_size) + (_f_mean * chunk_size)) / (est_size + chunk_size)
        f_std = ((f_std * est_size) + (_f_std * chunk_size)) / (est_size + chunk_size)

        est_size = est_size + chunk_size

    data._mmap.close()
    del data

    print(f"Training...", flush = True)

    stage = 0

    lr = 1e-6
    factor = 1.1
    interval = 10

    model = nn.Sequential(nn.Linear(_num_features, num_classes))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, nesterov = True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, min_lr = 1e-8, patience = args["patience_lr"] - 2, cooldown = 1)
    model.apply(init)
    model.train()

    stall_count = 0

    minibatch_count = 0
    best_validation_loss = np.inf
    stop = False

    best_loss_VA = np.inf

    state = {}

    for epoch in range(args["max_epochs"]):

        if epoch > 0 and stop:
            break

        chunks = torch.arange(training_size).split(args["chunk_size_sgd"])
        chunks = [chunks[_] for _ in torch.randperm(len(chunks))] # shuffle chunks

        for chunk_index, chunk in enumerate(chunks):

            if epoch > 0 and stop:
                break

            chunk_size = len(chunk)

            X_training_transform = cache_X[chunk]
            Y_training = cache_Y[chunk]

            minibatches = torch.randperm(chunk_size).split(args["minibatch_size"]) # shuffle within chunk

            for minibatch_index, minibatch in enumerate(minibatches):

                if epoch > 0 and stop:
                    break

                if minibatch_index > 0 and len(minibatch) < args["minibatch_size"]:
                    break

                X_mask = X_training_transform[minibatch] != 0

                optimizer.zero_grad()
                _Y_training = model(((X_training_transform[minibatch] - f_mean) * X_mask) / f_std)
                training_loss = loss_function(_Y_training, Y_training[minibatch])

                training_loss.backward()
                optimizer.step()

                minibatch_count += 1

                if stage == 0:

                    if minibatch_count % interval == 0:

                        with torch.no_grad():

                            model.eval()

                            _Y_validation = model(((X_validation_transform - f_mean) * validation_mask) / f_std)
                            validation_loss = loss_function(_Y_validation, Y_validation)

                            model.train()

                        if validation_loss.item() < best_loss_VA:
                            best_loss_VA = validation_loss.item()
                            state["model"] = copy.deepcopy(model.state_dict())
                            state["optim"] = copy.deepcopy(optimizer.state_dict())
                        elif validation_loss.item() > best_loss_VA:
                            stage = 1
                            model.load_state_dict(state["model"])
                            optimizer.load_state_dict(state["optim"])

                if stage == 0:

                    lr *= factor

                    for group in optimizer.param_groups:
                        group["lr"] = lr

        if stage == 1:

            with torch.no_grad():

                model.eval()

                _Y_validation = model(((X_validation_transform - f_mean) * validation_mask) / f_std)
                validation_loss = loss_function(_Y_validation, Y_validation)

                model.train()

            scheduler.step(validation_loss)

            if validation_loss.item() < best_validation_loss - args["threshold"]:
                best_validation_loss = validation_loss.item()
                best_model = copy.deepcopy(model)
                if not stop:
                    stall_count = 0
            else:
                stall_count += 1
                if stall_count >= args["patience"]:
                    stop = True
                    print(f"\n<Stopped at Epoch {epoch + 1}>")

    validation_accuracy = 0

    best_model.eval()

    if args["validate"]:
        with torch.no_grad():
            _Y_validation = best_model(((X_validation_transform - f_mean) * validation_mask) / f_std)
        validation_accuracy = (_Y_validation.argmax(-1) == Y_validation).numpy().mean()

    return transform, best_model, f_mean, f_std, validation_accuracy

def predict(path,
            transform,
            model,
            f_mean,
            f_std,
            **kwargs):

    args = \
    {
        "score"      : True,
        "batch_size" : 256,
        "test_size"  : None,
    }
    args = {**args, **kwargs}

    model.eval()

    data = np.load(path, mmap_mode = "r")

    max_size = data.shape[0]

    indices = torch.arange(max_size)

    batches = indices.split(args["batch_size"])

    predictions = []

    correct = 0
    total = 0

    print("Predicting...")

    for batch_index, batch in enumerate(batches):

        test_data = torch.tensor(data[batch])
        X_test, Y_test = test_data[:, :-1].float().unsqueeze(1), test_data[:, -1].long()

        X_test_transform = transform(X_test).clamp(0).sqrt()

        X_mask = X_test_transform != 0

        X_test_transform = ((X_test_transform - f_mean) * X_mask) / f_std

        with torch.no_grad():
            _predictions = model(X_test_transform).argmax(1)
        predictions.append(_predictions)

        total += len(test_data)
        correct += (_predictions == Y_test).long().sum()

    data._mmap.close()
    del data

    if args["score"]:
        return np.concatenate(predictions), correct / total
    else:
        return np.concatenate(predictions)
