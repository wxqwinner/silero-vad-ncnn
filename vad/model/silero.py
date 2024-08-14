import pnnx
import torch
from vad.model.nets.silero import SileroNet

MAP_TABLE = {
'adaptive_normalization.filter_'                : 'adaptive_normalization.filter_', # torch.Size([1, 1, 7])
'feature_extractor.forward_basis_buffer'        : 'feature_extractor.forward_basis_buffer', # torch.Size([258, 1, 256])
'first_layer.conv1.weight'                      : 'first_layer.0.dw_conv.0.weight', # torch.Size([258, 1, 5])
'first_layer.conv1.bias'                        : 'first_layer.0.dw_conv.0.bias', # torch.Size([258])
'first_layer.conv2.weight'                      : 'first_layer.0.pw_conv.0.weight', # torch.Size([16, 258, 1])
'first_layer.conv2.bias'                        : 'first_layer.0.pw_conv.0.bias', # torch.Size([16])
'first_layer.conv3.weight'                      : 'first_layer.0.proj.weight', # torch.Size([16, 258, 1])
'first_layer.conv3.bias'                        : 'first_layer.0.proj.bias', # torch.Size([16])
'encoder.conv_block1.conv1.weight'              : 'encoder.0.weight', # torch.Size([16, 16, 1])
'encoder.conv_block1.conv1.bias'                : 'encoder.0.bias', # torch.Size([16])
'encoder.conv_block1.bn1.weight'                : 'encoder.1.weight', # torch.Size([16])
'encoder.conv_block1.bn1.bias'                  : 'encoder.1.bias', # torch.Size([16])
'encoder.conv_block1.bn1.running_mean'          : 'encoder.1.running_mean', # torch.Size([16])
'encoder.conv_block1.bn1.running_var'           : 'encoder.1.running_var', # torch.Size([16])
'encoder.conv_block1.conv2.weight'              : 'encoder.3.0.dw_conv.0.weight', # torch.Size([16, 1, 5])'
'encoder.conv_block1.conv2.bias'                : 'encoder.3.0.dw_conv.0.bias',  #torch.Size([16])
'encoder.conv_block1.conv3.weight'              : 'encoder.3.0.pw_conv.0.weight',  #torch.Size([32, 16, 1])
'encoder.conv_block1.conv3.bias'                : 'encoder.3.0.pw_conv.0.bias',  #torch.Size([32])
'encoder.conv_block1.conv4.weight'              : 'encoder.3.0.proj.weight',  #torch.Size([32, 16, 1])
'encoder.conv_block1.conv4.bias'                : 'encoder.3.0.proj.bias',  #torch.Size([32])
'encoder.conv_block1.conv5.weight'              : 'encoder.4.weight',  #torch.Size([32, 32, 1])
'encoder.conv_block1.conv5.bias'                : 'encoder.4.bias',  #torch.Size([32])
'encoder.conv_block1.bn2.weight'                : 'encoder.5.weight',  #torch.Size([32])
'encoder.conv_block1.bn2.bias'                  : 'encoder.5.bias',  #torch.Size([32])
'encoder.conv_block1.bn2.running_mean'          : 'encoder.5.running_mean',  #torch.Size([32])
'encoder.conv_block1.bn2.running_var'           : 'encoder.5.running_var',  #torch.Size([32])
'encoder.conv_block2.conv1.weight'              : 'encoder.8.weight',  #torch.Size([32, 32, 1])
'encoder.conv_block2.conv1.bias'                : 'encoder.8.bias',  #torch.Size([32])
'encoder.conv_block2.bn1.weight'                : 'encoder.9.weight',  #torch.Size([32])
'encoder.conv_block2.bn1.bias'                  : 'encoder.9.bias',  #torch.Size([32])
'encoder.conv_block2.bn1.running_mean'          : 'encoder.9.running_mean',  #torch.Size([32])
'encoder.conv_block2.bn1.running_var'           : 'encoder.9.running_var',  #torch.Size([32])
'encoder.conv_block2.conv2.weight'              : 'encoder.11.0.dw_conv.0.weight',  #torch.Size([32, 1, 5])
'encoder.conv_block2.conv2.bias'                : 'encoder.11.0.dw_conv.0.bias',  #torch.Size([32])
'encoder.conv_block2.conv3.weight'              : 'encoder.11.0.pw_conv.0.weight',  #torch.Size([64, 32, 1])
'encoder.conv_block2.conv3.bias'                : 'encoder.11.0.pw_conv.0.bias',  #torch.Size([64])
'encoder.conv_block2.conv4.weight'              : 'encoder.11.0.proj.weight',  #torch.Size([64, 32, 1])
'encoder.conv_block2.conv4.bias'                : 'encoder.11.0.proj.bias',  #torch.Size([64])
'encoder.conv_block2.conv5.weight'              : 'encoder.12.weight',  #torch.Size([64, 64, 1])
'encoder.conv_block2.conv5.bias'                : 'encoder.12.bias',  #torch.Size([64])
'encoder.conv_block2.bn2.weight'                : 'encoder.13.weight',  #torch.Size([64])
'encoder.conv_block2.bn2.bias'                  : 'encoder.13.bias',  #torch.Size([64])
'encoder.conv_block2.bn2.running_mean'          : 'encoder.13.running_mean',  #torch.Size([64])
'encoder.conv_block2.bn2.running_var'           : 'encoder.13.running_var',  #torch.Size([64])
'encoder.conv1.weight'                          : 'encoder.7.0.dw_conv.0.weight',  #torch.Size([32, 1, 5])
'encoder.conv1.bias'                            : 'encoder.7.0.dw_conv.0.bias',  #torch.Size([32])
'encoder.conv2.weight'                          : 'encoder.7.0.pw_conv.0.weight',  #torch.Size([32, 32, 1])
'encoder.conv2.bias'                            : 'encoder.7.0.pw_conv.0.bias',  #torch.Size([32])
'decoder.lstm1.weight_ih_l0'                    : 'decoder.rnn.weight_ih_l0', #torch.Size([256, 64])
'decoder.lstm1.weight_hh_l0'                    : 'decoder.rnn.weight_hh_l0', #torch.Size([256, 64])
'decoder.lstm1.bias_ih_l0'                      : 'decoder.rnn.bias_ih_l0', #torch.Size([256])
'decoder.lstm1.bias_hh_l0'                      : 'decoder.rnn.bias_hh_l0', #torch.Size([256])
'decoder.lstm1.weight_ih_l1'                    : 'decoder.rnn.weight_ih_l1', #torch.Size([256, 64])
'decoder.lstm1.weight_hh_l1'                    : 'decoder.rnn.weight_hh_l1', #torch.Size([256, 64])
'decoder.lstm1.bias_ih_l1'                      : 'decoder.rnn.bias_ih_l1', #torch.Size([256])
'decoder.lstm1.bias_hh_l1'                      : 'decoder.rnn.bias_hh_l1', #torch.Size([256])
'decoder.conv1.weight'                          : 'decoder.decoder.1.weight', #torch.Size([1, 64, 1])
'decoder.conv1.bias'                            : 'decoder.decoder.1.bias', #torch.Size([1])
}


class SileroModel():
    def __init__(self):
        self.net = SileroNet()

    def convert(self, model_in):
        """from external model import"""
        torch.set_grad_enabled(False)
        model = torch.jit.load(model_in, map_location='cpu')
        model.eval()
        jit_net = model._model # only 16k
        jit_net_dict = jit_net.state_dict()

        x = torch.rand(512).reshape(1, -1)
        h = torch.rand(2, 1, 64)
        c = torch.rand(2, 1, 64)

        self.net.eval()
        net_dict = self.net.state_dict()

        for v in net_dict:
            print(v)
            if v in MAP_TABLE:
                net_dict[v] = jit_net_dict[MAP_TABLE[v]]

        self.net.load_state_dict(net_dict, strict=False)

        mod = torch.jit.trace(self.net, (x, h, c))
        mod.save('silero.jit')
        import pnnx
        # pnnx.export(self.net, 'silero.pt', (x, h, c), check_trace=False)
        pnnx.convert('silero.jit', (x, h, c))
