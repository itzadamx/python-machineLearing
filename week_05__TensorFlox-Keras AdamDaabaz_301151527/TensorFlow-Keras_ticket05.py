Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import tensorflow as tf



import pprint
pprint.pprint(dir(tf))
['AggregationMethod',
 'Assert',
 'CriticalSection',
 'DType',
 'DeviceSpec',
 'GradientTape',
 'Graph',
 'IndexedSlices',
 'IndexedSlicesSpec',
 'Module',
 'Operation',
 'OptionalSpec',
 'RaggedTensor',
 'RaggedTensorSpec',
 'RegisterGradient',
 'SparseTensor',
 'SparseTensorSpec',
 'Tensor',
 'TensorArray',
 'TensorArraySpec',
 'TensorShape',
 'TensorSpec',
 'TypeSpec',
 'UnconnectedGradients',
 'Variable',
 'VariableAggregation',
 'VariableSynchronization',
 '_API_MODULE',
 '_LazyLoader',
 '__all__',
 '__builtins__',
 '__cached__',
 '__compiler_version__',
 '__cxx11_abi_flag__',
 '__doc__',
 '__file__',
 '__git_version__',
 '__internal__',
 '__loader__',
 '__monolithic_build__',
 '__name__',
 '__operators__',
 '__package__',
 '__path__',
 '__spec__',
 '__version__',
 '__warningregistry__',
 '_api',
 '_compat',
 '_current_file_location',
 '_current_module',
 '_distutils',
 '_estimator_module',
 '_fi',
 '_inspect',
 '_keras_module',
 '_keras_package',
 '_kernel_dir',
 '_ll',
 '_logging',
 '_major_api_version',
 '_module_dir',
 '_module_util',
 '_names_with_underscore',
 '_os',
 '_plugin_dir',
 '_running_from_pip_package',
 '_s',
 '_site',
 '_site_packages_dirs',
 '_sys',
 '_tf2',
 '_tf_api_dir',
 '_tf_dir',
 '_typing',
 'abs',
 'acos',
 'acosh',
 'add',
 'add_n',
 'argmax',
 'argmin',
 'argsort',
 'as_dtype',
 'as_string',
 'asin',
 'asinh',
 'assert_equal',
 'assert_greater',
 'assert_less',
 'assert_rank',
 'atan',
 'atan2',
 'atanh',
 'audio',
 'autodiff',
 'autograph',
 'batch_to_space',
 'bfloat16',
 'bitcast',
 'bitwise',
 'bool',
 'boolean_mask',
 'broadcast_dynamic_shape',
 'broadcast_static_shape',
 'broadcast_to',
 'case',
 'cast',
 'clip_by_global_norm',
 'clip_by_norm',
 'clip_by_value',
 'compat',
 'complex',
 'complex128',
 'complex64',
 'concat',
 'cond',
 'config',
 'constant',
 'constant_initializer',
 'control_dependencies',
 'convert_to_tensor',
 'cos',
 'cosh',
 'cumsum',
 'custom_gradient',
 'data',
 'debugging',
 'device',
 'distribute',
 'divide',
 'double',
 'dtypes',
 'dynamic_partition',
 'dynamic_stitch',
 'edit_distance',
 'eig',
 'eigvals',
 'einsum',
 'ensure_shape',
 'equal',
 'errors',
 'estimator',
 'executing_eagerly',
 'exp',
 'expand_dims',
 'experimental',
 'extract_volume_patches',
 'eye',
 'feature_column',
 'fill',
 'fingerprint',
 'float16',
 'float32',
 'float64',
 'floor',
 'foldl',
 'foldr',
 'function',
 'gather',
 'gather_nd',
 'get_current_name_scope',
 'get_logger',
 'get_static_value',
 'grad_pass_through',
 'gradients',
 'graph_util',
 'greater',
 'greater_equal',
 'group',
 'guarantee_const',
 'half',
 'hessians',
 'histogram_fixed_width',
 'histogram_fixed_width_bins',
 'identity',
 'identity_n',
 'image',
 'import_graph_def',
 'init_scope',
 'initializers',
 'inside_function',
 'int16',
 'int32',
 'int64',
 'int8',
 'io',
 'is_tensor',
 'keras',
 'less',
 'less_equal',
 'linalg',
 'linspace',
 'lite',
 'load_library',
 'load_op_library',
 'logical_and',
 'logical_not',
 'logical_or',
 'lookup',
 'losses',
 'make_ndarray',
 'make_tensor_proto',
 'map_fn',
 'math',
 'matmul',
 'matrix_square_root',
 'maximum',
 'meshgrid',
 'metrics',
 'minimum',
 'mixed_precision',
 'mlir',
 'multiply',
 'name_scope',
 'negative',
 'nest',
 'newaxis',
 'nn',
 'no_gradient',
 'no_op',
 'nondifferentiable_batch_function',
 'norm',
 'not_equal',
 'numpy_function',
 'one_hot',
 'ones',
 'ones_initializer',
 'ones_like',
 'optimizers',
 'pad',
 'parallel_stack',
 'pow',
 'print',
 'profiler',
 'py_function',
 'qint16',
 'qint32',
 'qint8',
 'quantization',
 'queue',
 'quint16',
 'quint8',
 'ragged',
 'random',
 'random_normal_initializer',
 'random_uniform_initializer',
 'range',
 'rank',
 'raw_ops',
 'realdiv',
 'recompute_grad',
 'reduce_all',
 'reduce_any',
 'reduce_logsumexp',
 'reduce_max',
 'reduce_mean',
 'reduce_min',
 'reduce_prod',
 'reduce_sum',
 'register_tensor_conversion_function',
 'repeat',
 'required_space_to_batch_paddings',
 'reshape',
 'resource',
 'reverse',
 'reverse_sequence',
 'roll',
 'round',
 'saturate_cast',
 'saved_model',
 'scalar_mul',
 'scan',
 'scatter_nd',
 'searchsorted',
 'sequence_mask',
 'sets',
 'shape',
 'shape_n',
 'sigmoid',
 'sign',
 'signal',
 'sin',
 'sinh',
 'size',
 'slice',
 'sort',
 'space_to_batch',
 'space_to_batch_nd',
 'sparse',
 'split',
 'sqrt',
 'square',
 'squeeze',
 'stack',
 'stop_gradient',
 'strided_slice',
 'string',
 'strings',
 'subtract',
 'summary',
 'switch_case',
 'sysconfig',
 'tan',
 'tanh',
 'tensor_scatter_nd_add',
 'tensor_scatter_nd_max',
 'tensor_scatter_nd_min',
 'tensor_scatter_nd_sub',
 'tensor_scatter_nd_update',
 'tensordot',
 'test',
 'tile',
 'timestamp',
 'tools',
 'tpu',
 'train',
 'transpose',
 'truediv',
 'truncatediv',
 'truncatemod',
 'tuple',
 'type_spec_from_value',
 'types',
 'uint16',
 'uint32',
 'uint64',
 'uint8',
 'unique',
 'unique_with_counts',
 'unravel_index',
 'unstack',
 'variable_creator_scope',
 'variant',
 'vectorized_map',
 'version',
 'where',
 'while_loop',
 'xla',
 'zeros',
 'zeros_initializer',
 'zeros_like']
pprint.pprint(dir(tf.keras))
['Input',
 'Model',
 'Sequential',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__internal__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 '__version__',
 '_sys',
 'activations',
 'applications',
 'backend',
 'callbacks',
 'constraints',
 'datasets',
 'estimator',
 'experimental',
 'initializers',
 'layers',
 'losses',
 'metrics',
 'mixed_precision',
 'models',
 'optimizers',
 'preprocessing',
 'regularizers',
 'utils',
 'wrappers']
x=tf.constant([[5,2],[1,3]])
x
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[5, 2],
       [1, 3]])>
print(x)
tf.Tensor(
[[5 2]
 [1 3]], shape=(2, 2), dtype=int32)
x.numpy()
array([[5, 2],
       [1, 3]])
x.dtype
tf.int32
x.shape
TensorShape([2, 2])
x.backing_device
'/job:localhost/replica:0/task:0/device:CPU:0'



tf.ones(shape=(3,3))
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]], dtype=float32)>


tf.random.normal(shape=(2,3))
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[-0.04267993, -0.21212167,  0.3684062 ],
       [-0.2898761 ,  0.4127003 ,  1.4572965 ]], dtype=float32)>


tf.random.normal(shape=(2,3),mean =5 , stddev=3)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[5.4505367, 3.5271738, 9.109539 ],
       [5.7440243, 3.9143407, 7.887212 ]], dtype=float32)>
'DType',
('DType',)













xvar=tf.random.normal(shape=(3,3))
xvar
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[-1.3431174 , -1.2798631 ,  1.0963361 ],
       [-2.0625205 ,  0.9161497 , -0.10989331],
       [-0.5237184 ,  1.2490162 ,  0.447295  ]], dtype=float32)>
a=tf.Variable(xvar)
a
<tf.Variable 'Variable:0' shape=(3, 3) dtype=float32, numpy=
array([[-1.3431174 , -1.2798631 ,  1.0963361 ],
       [-2.0625205 ,  0.9161497 , -0.10989331],
       [-0.5237184 ,  1.2490162 ,  0.447295  ]], dtype=float32)>
a.assign(tf.random.normal(shape=(3,3)))
<tf.Variable 'UnreadVariable' shape=(3, 3) dtype=float32, numpy=
array([[ 2.1477382 ,  0.26799202,  0.47285423],
       [ 1.3153702 ,  1.1347489 ,  1.2839893 ],
       [-0.753463  ,  0.86000854, -0.21058422]], dtype=float32)>
xvar_t_add=tf.random.normal(shape=(3,3))
xvar_t_add
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 0.6912952 ,  0.69325733, -1.1165934 ],
       [-0.29811534,  1.0890328 ,  1.4640796 ],
       [-0.24748404, -0.6215593 , -1.0459993 ]], dtype=float32)>
a.assign_add(xvar_t_add)
<tf.Variable 'UnreadVariable' shape=(3, 3) dtype=float32, numpy=
array([[ 2.8390334 ,  0.96124935, -0.6437391 ],
       [ 1.0172548 ,  2.2237816 ,  2.7480688 ],
       [-1.000947  ,  0.23844922, -1.2565835 ]], dtype=float32)>
a.assign_sub(xvar_t_add)
<tf.Variable 'UnreadVariable' shape=(3, 3) dtype=float32, numpy=
array([[ 2.1477382 ,  0.26799202,  0.47285426],
       [ 1.3153702 ,  1.1347488 ,  1.2839892 ],
       [-0.753463  ,  0.86000854, -0.21058416]], dtype=float32)>




new_value=tf.random.normal(shape=(3,3))
new_value
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 0.674638  ,  0.33250847,  0.5552147 ],
       [ 0.13282615,  0.6068088 , -1.8776883 ],
       [-0.651278  , -0.12238377,  0.41858813]], dtype=float32)>
a
<tf.Variable 'Variable:0' shape=(3, 3) dtype=float32, numpy=
array([[ 2.1477382 ,  0.26799202,  0.47285426],
       [ 1.3153702 ,  1.1347488 ,  1.2839892 ],
       [-0.753463  ,  0.86000854, -0.21058416]], dtype=float32)>
a.assign_add(new_value)
<tf.Variable 'UnreadVariable' shape=(3, 3) dtype=float32, numpy=
array([[ 2.8223763 ,  0.60050046,  1.028069  ],
       [ 1.4481964 ,  1.7415576 , -0.5936991 ],
       [-1.404741  ,  0.73762476,  0.20800397]], dtype=float32)>
b=tf.random.normal(shape=(3,3))
b
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[-0.00207813, -0.21755147,  0.9758152 ],
       [-0.16946384,  1.5797902 ,  0.6808079 ],
       [-0.77444243,  0.05250808,  1.1428487 ]], dtype=float32)>
c=a+b
c
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 2.8202982 ,  0.382949  ,  2.0038843 ],
       [ 1.2787325 ,  3.3213477 ,  0.08710879],
       [-2.1791835 ,  0.7901328 ,  1.3508527 ]], dtype=float32)>
d=tf.square(c)
d
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[7.9540820e+00, 1.4664993e-01, 4.0155525e+00],
       [1.6351569e+00, 1.1031351e+01, 7.5879414e-03],
       [4.7488408e+00, 6.2430990e-01, 1.8248031e+00]], dtype=float32)>
e=tf.exp(c)
e
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[16.781855  ,  1.4666033 ,  7.4178133 ],
       [ 3.592084  , 27.697653  ,  1.0910155 ],
       [ 0.11313387,  2.203689  ,  3.8607163 ]], dtype=float32)>
with tf.GradientTape() as tape:
    tape.watch(a)
    f=tf.sqrt(tf.square(a)+tf.square(b))df_da=tape.gradient(f,a)
    
SyntaxError: invalid syntax
with tf.GradientTape() as tape:
    tape.watch(a)
    f=tf.sqrt(tf.square(a)+tf.square(b))
    df_da=tape.gradient(f,a)
    print(df_da)

    
tf.Tensor(
[[ 0.9999997   0.94020134  0.7252989 ]
 [ 0.9932231   0.74066865 -0.65724486]
 [-0.8757324   0.9974759   0.17906317]], shape=(3, 3), dtype=float32)
