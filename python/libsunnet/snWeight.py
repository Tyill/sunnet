import h5py
import numpy as np
import keras.utils as keras_utils

RESNET50_W_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')

def _load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    # Returns
        data: Attributes data.
    """
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                         for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data


def getResNet50Weights():
    """Implements topological (order-based) weight loading.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """

    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')

    weights_path = keras_utils.get_file(
        'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        RESNET50_W_PATH,
        cache_subdir='models',
        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')

    with h5py.File(weights_path, mode='r') as f:

        layer_names = _load_attributes_from_hdf5_group(f, 'layer_names')
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = _load_attributes_from_hdf5_group(g, 'weight_names')
            if weight_names:
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names

        weight_values = {}
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = _load_attributes_from_hdf5_group(g, 'weight_names')
            weight_values[name] = [np.asarray(g[weight_name]) for weight_name in weight_names]

    return weight_values
