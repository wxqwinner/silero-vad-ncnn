import os
import ncnn
import numpy as np

class SileroNCNN():
    def __init__(self, model_path=None):
        cwd_dir = os.path.split(os.path.realpath(__file__))[0]
        net = ncnn.Net()

        if model_path:
            net.load_param(f'{model_path}.param')
            net.load_model(f'{model_path}.bin')
        else:
            net.load_param(f'{cwd_dir}/resources/silero.ncnn.param')
            net.load_model(f'{cwd_dir}/resources/silero.ncnn.bin')
        self.h = np.zeros((2, 1, 64,), dtype=np.float32)
        self.c = np.zeros((2, 1, 64,), dtype=np.float32)
        self.net = net

    def inference(self, x):
        x = x.reshape(1, 512)
        h = self.h
        c = self.c

        for i in range(x.shape[0]):
            in0 = x[i, :].reshape(1, -1)
            ex = self.net.create_extractor()
            ex.input('in0', ncnn.Mat(in0.squeeze(0)).clone())
            ex.input('in1', ncnn.Mat(h).clone())
            ex.input('in2', ncnn.Mat(c).clone())
            _, out0 = ex.extract('out0')
            _, out1 = ex.extract('out1')
            _, out2 = ex.extract('out2')
            h = np.array(out1)
            c = np.array(out2)

        self.h = h
        self.c = c

        return out0[0]