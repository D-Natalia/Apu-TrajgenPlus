import os
import pickle
import numpy as np
base = r'D:\Work\Licenta\APU-TrajGenPlus-main'
pkl_path = os.path.join(base, 'models', 'mdlgru-porto.pkl')
print('PKL exists', os.path.exists(pkl_path))
with open(pkl_path, 'rb') as f:
    model = pickle.load(f)
print('type:', type(model))
try:
    print('output_shape:', model.output_shape)
except Exception as e:
    print('output_shape error:', repr(e))
try:
    model.summary()
except Exception as e:
    print('summary error:', repr(e))
try:
    x = np.zeros((1,25,3), dtype=np.float32)
    y = model.predict(x, batch_size=1, verbose=0)
    print('predict shape:', np.shape(y), 'dtype:', y.dtype)
    print('predict min/max/mean:', np.min(y), np.max(y), np.mean(y))
    print('predict sample:', y.flatten()[:20])
except Exception as e:
    print('predict error:', repr(e))
