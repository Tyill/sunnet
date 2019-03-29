import h5py
import numpy as np

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

def _preprocess_weights_for_loading(weights):


    return weights
#     """Converts layers weights from Keras 1 format to Keras 2.
#
#     # Arguments
#         layer: Layer instance.
#         weights: List of weights values (Numpy arrays).
#         original_keras_version: Keras version for the weights, as a string.
#         original_backend: Keras backend the weights were trained with,
#             as a string.
#         reshape: Reshape weights to fit the layer when the correct number
#             of values are present but the shape does not match.
#
#     # Returns
#         A list of weights values (Numpy arrays).
#     """
#     def convert_nested_bidirectional(weights):
#         """Converts layers nested in `Bidirectional` wrapper.
#
#         # Arguments
#             weights: List of weights values (Numpy arrays).
#         # Returns
#             A list of weights values (Numpy arrays).
#         """
#         num_weights_per_layer = len(weights) // 2
#         forward_weights = preprocess_weights_for_loading(
#             layer.forward_layer,
#             weights[:num_weights_per_layer],
#             original_keras_version,
#             original_backend)
#         backward_weights = preprocess_weights_for_loading(
#             layer.backward_layer,
#             weights[num_weights_per_layer:],
#             original_keras_version,
#             original_backend)
#         return forward_weights + backward_weights
#
#     def convert_nested_time_distributed(weights):
#         """Converts layers nested in `TimeDistributed` wrapper.
#
#         # Arguments
#             weights: List of weights values (Numpy arrays).
#         # Returns
#             A list of weights values (Numpy arrays).
#         """
#         return preprocess_weights_for_loading(
#             layer.layer, weights, original_keras_version, original_backend)
#
#     def convert_nested_model(weights):
#         """Converts layers nested in `Model` or `Sequential`.
#
#         # Arguments
#             weights: List of weights values (Numpy arrays).
#         # Returns
#             A list of weights values (Numpy arrays).
#         """
#         new_weights = []
#         # trainable weights
#         for sublayer in layer.layers:
#             num_weights = len(sublayer.trainable_weights)
#             if num_weights > 0:
#                 new_weights.extend(preprocess_weights_for_loading(
#                     layer=sublayer,
#                     weights=weights[:num_weights],
#                     original_keras_version=original_keras_version,
#                     original_backend=original_backend))
#                 weights = weights[num_weights:]
#
#         # non-trainable weights
#         for sublayer in layer.layers:
#             num_weights = len([l for l in sublayer.weights
#                                if l not in sublayer.trainable_weights])
#             if num_weights > 0:
#                 new_weights.extend(preprocess_weights_for_loading(
#                     layer=sublayer,
#                     weights=weights[:num_weights],
#                     original_keras_version=original_keras_version,
#                     original_backend=original_backend))
#                 weights = weights[num_weights:]
#         return new_weights
#
#     # Convert layers nested in Bidirectional/TimeDistributed/Model/Sequential.
#     # Both transformation should be ran for both Keras 1->2 conversion
#     # and for conversion of CuDNN layers.
#     if layer.__class__.__name__ == 'Bidirectional':
#         weights = convert_nested_bidirectional(weights)
#     if layer.__class__.__name__ == 'TimeDistributed':
#         weights = convert_nested_time_distributed(weights)
#     elif layer.__class__.__name__ in ['Model', 'Sequential']:
#         weights = convert_nested_model(weights)
#
#     if original_keras_version == '1':
#         if layer.__class__.__name__ == 'TimeDistributed':
#             weights = preprocess_weights_for_loading(layer.layer,
#                                                      weights,
#                                                      original_keras_version,
#                                                      original_backend)
#
#         if layer.__class__.__name__ == 'Conv1D':
#             shape = weights[0].shape
#             # Handle Keras 1.1 format
#             if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
#                 # Legacy shape:
#                 # (filters, input_dim, filter_length, 1)
#                 assert (shape[0] == layer.filters and
#                         shape[2:] == (layer.kernel_size[0], 1))
#                 weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
#             weights[0] = weights[0][:, 0, :, :]
#
#         if layer.__class__.__name__ == 'Conv2D':
#             if layer.data_format == 'channels_first':
#                 # old: (filters, stack_size, kernel_rows, kernel_cols)
#                 # new: (kernel_rows, kernel_cols, stack_size, filters)
#                 weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
#
#         if layer.__class__.__name__ == 'Conv2DTranspose':
#             if layer.data_format == 'channels_last':
#                 # old: (kernel_rows, kernel_cols, stack_size, filters)
#                 # new: (kernel_rows, kernel_cols, filters, stack_size)
#                 weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
#             if layer.data_format == 'channels_first':
#                 # old: (filters, stack_size, kernel_rows, kernel_cols)
#                 # new: (kernel_rows, kernel_cols, filters, stack_size)
#                 weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
#
#         if layer.__class__.__name__ == 'Conv3D':
#             if layer.data_format == 'channels_first':
#                 # old: (filters, stack_size, ...)
#                 # new: (..., stack_size, filters)
#                 weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))
#
#         if layer.__class__.__name__ == 'GRU':
#             if len(weights) == 9:
#                 kernel = np.concatenate([weights[0],
#                                          weights[3],
#                                          weights[6]], axis=-1)
#                 recurrent_kernel = np.concatenate([weights[1],
#                                                    weights[4],
#                                                    weights[7]], axis=-1)
#                 bias = np.concatenate([weights[2],
#                                        weights[5],
#                                        weights[8]], axis=-1)
#                 weights = [kernel, recurrent_kernel, bias]
#
#         if layer.__class__.__name__ == 'LSTM':
#             if len(weights) == 12:
#                 # old: i, c, f, o
#                 # new: i, f, c, o
#                 kernel = np.concatenate([weights[0],
#                                          weights[6],
#                                          weights[3],
#                                          weights[9]], axis=-1)
#                 recurrent_kernel = np.concatenate([weights[1],
#                                                    weights[7],
#                                                    weights[4],
#                                                    weights[10]], axis=-1)
#                 bias = np.concatenate([weights[2],
#                                        weights[8],
#                                        weights[5],
#                                        weights[11]], axis=-1)
#                 weights = [kernel, recurrent_kernel, bias]
#
#         if layer.__class__.__name__ == 'ConvLSTM2D':
#             if len(weights) == 12:
#                 kernel = np.concatenate([weights[0],
#                                          weights[6],
#                                          weights[3],
#                                          weights[9]], axis=-1)
#                 recurrent_kernel = np.concatenate([weights[1],
#                                                    weights[7],
#                                                    weights[4],
#                                                    weights[10]], axis=-1)
#                 bias = np.concatenate([weights[2],
#                                        weights[8],
#                                        weights[5],
#                                        weights[11]], axis=-1)
#                 if layer.data_format == 'channels_first':
#                     # old: (filters, stack_size, kernel_rows, kernel_cols)
#                     # new: (kernel_rows, kernel_cols, stack_size, filters)
#                     kernel = np.transpose(kernel, (2, 3, 1, 0))
#                     recurrent_kernel = np.transpose(recurrent_kernel,
#                                                     (2, 3, 1, 0))
#                 weights = [kernel, recurrent_kernel, bias]
#
#     conv_layers = ['Conv1D',
#                    'Conv2D',
#                    'Conv3D',
#                    'Conv2DTranspose',
#                    'ConvLSTM2D']
#     if layer.__class__.__name__ in conv_layers:
#         layer_weights_shape = K.int_shape(layer.weights[0])
#         if _need_convert_kernel(original_backend):
#             weights[0] = conv_utils.convert_kernel(weights[0])
#             if layer.__class__.__name__ == 'ConvLSTM2D':
#                 weights[1] = conv_utils.convert_kernel(weights[1])
#         if reshape and layer_weights_shape != weights[0].shape:
#             if weights[0].size != np.prod(layer_weights_shape):
#                 raise ValueError('Weights must be of equal size to ' +
#                                  'apply a reshape operation. ' +
#                                  'Layer ' + layer.name +
#                                  '\'s weights have shape ' +
#                                  str(layer_weights_shape) + ' and size ' +
#                                  str(np.prod(layer_weights_shape)) + '. ' +
#                                  'The weights for loading have shape ' +
#                                  str(weights[0].shape) + ' and size ' +
#                                  str(weights[0].size) + '. ')
#             weights[0] = np.reshape(weights[0], layer_weights_shape)
#         elif layer_weights_shape != weights[0].shape:
#             weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
#             if layer.__class__.__name__ == 'ConvLSTM2D':
#                 weights[1] = np.transpose(weights[1], (3, 2, 0, 1))
#
#     # convert CuDNN layers
#     weights = _convert_rnn_weights(layer, weights)
#
#     return weights

def loadHdf5Group(filepath):
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

    with h5py.File(filepath, mode='r') as f:

        layer_names = _load_attributes_from_hdf5_group(f, 'layer_names')
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = _load_attributes_from_hdf5_group(g, 'weight_names')
            if weight_names:
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names

        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = _load_attributes_from_hdf5_group(g, 'weight_names')
            weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

            weight_values = _preprocess_weights_for_loading(weight_values)
            weight_value_tuples.append((name, weight_values))

    return weight_value_tuples
