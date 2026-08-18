import torch

def position(position, features, layers, activation):
    """Use the position as input (i.e. no conditioning)
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = position

    for i, l in enumerate(layers):
        h = activation(l(h))

    return h


def feature(position, features, layers, activation):
    """Use the features as input.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = features

    for i, l in enumerate(layers):
        h = activation(l(h))

    return h


def concat(position, features, layers, activation):
    """Concatenates the input onto the features, and then feeds into the input of the neural network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = torch.cat([position, features], dim=-1)

    for i, l in enumerate(layers):
        h = activation(l(h))
    return h


def film_linear(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (multiply only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong
    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):
        # Maybe also add another assertion here
        h = activation(l(h) * features[..., i, :])
    return h


def film_translate(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]  # features.shape[:-1]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong

    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):

        #print(f"l(h) is {l(h)}")
        #print(f"modulation is  is {features[..., i, :]}")
        # Maybe also add another assertion here
        h = activation(l(h) + features[..., i, :])
    return h


def film(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong
    
    if with_batch:
        features = features.reshape(
        feature_shape, 1, 2, num_hidden, feature_dim // (num_hidden * 2)
    )
    else:

        features = features.reshape(
            feature_shape, 2, num_hidden, feature_dim // (num_hidden * 2)
        )

    h = position

    for i, l in enumerate(layers):
        # Maybe also add another assertion here
        h = activation(l(h) * (1+ features[..., 0, i, :]) + features[..., 1, i, :])
    return h


def film_translate_with_perturbation(position, features, layers, activation, with_batch=True, layer_index=4, channel_index=124, noise_level=1):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    feature_shape = features.shape[0]  # features.shape[:-1]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)

    #assert layer_index < num_hidden and channel_index < feature_dim


    # Maybe add assertion here... but if it errors, your feature_dim size is wrong

    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    perturbation = torch.zeros_like(features)
    #perturbation[..., layer_index, :] = torch.randn(feature_shape, feature_dim // num_hidden) * noise_level
    perturbation[..., layer_index, channel_index] = perturbation[..., layer_index, channel_index] * noise_level + 0.01

    h = position

    for i, l in enumerate(layers):

        print(f"l(h) is {l(h)}")
        print(f"modulation is  is {features[..., i, :]}")
        # Maybe also add another assertion here
        h = activation(l(h) + features[..., i, :] + perturbation[..., i, :])
    return h


def film_translate_per_layers(position, features, layers, activation):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = position

    for i, l in enumerate(layers):
        # Maybe also add another assertion here
        print(f"l(h) is {l(h)}")
        print(f"modulation is  is {features[..., i, :]}")
        h = activation(l(h) + features[..., i, :])
    return h