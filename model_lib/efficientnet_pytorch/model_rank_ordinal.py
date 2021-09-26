"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from model_lib.efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)
from torchsummary import summary

class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N x 2K x H x W; N - batch_size, 2K - channels, K - number of discrete sub-intervals
        :return:  labels - ordinal labels (corresponding to discrete depth values) of size N x 1 x H x W
                  softmax - predicted softmax probabilities P (as in the paper) of size N x K x H x W
        """
        N, K= x.size()
        K = K // 2  # number of discrete sub-intervals

        odd = x[:, ::2].clone()
        even = x[:, 1::2].clone()

        odd = odd.view(N, 1, K)
        even = even.view(N, 1, K)

        paired_channels = torch.cat((odd, even), dim=1)
        paired_channels = paired_channels.clamp(min=1e-8, max=1e8)  # prevent nans

        softmax = F.softmax(paired_channels, dim=1)

        softmax = softmax[:, 1, :]
        softmax = softmax.view(-1, K)
        labels = torch.sum((softmax > 0.5), dim=1).view(-1, 1) - 1
        return labels[:, 0], softmax


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    # Example:
    #     >>> import torch
    #     >>> from efficientnet.model import EfficientNet
    #     >>> inputs = torch.rand(1, 3, 224, 224)
    #     >>> model = EfficientNet.from_pretrained('efficientnet-b0')
    #     >>> model.eval()
    #     >>> outputs = model(inputs)
    """

    def __init__(self, task_mode='class', blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.task_mode = task_mode

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self._dropout = nn.Dropout(self._global_params.dropout_rate)
        # self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        # building classifier
        if self.task_mode in ['class', 'multi']:
            self.classifier_ = nn.Sequential(
                nn.Dropout(self._global_params.dropout_rate),
                nn.Linear(out_channels, self._global_params.num_classes),
            )
        if self.task_mode in ['regress', 'multi']:
            self.regressioner_ = nn.Sequential(
                nn.Dropout(self._global_params.dropout_rate),
                nn.Linear(out_channels, 1),
            )
        if self.task_mode in ['regress_rank_ordinal',]:
            self.regressioner_ = nn.Sequential(
                nn.Dropout(self._global_params.dropout_rate),
                nn.Linear(out_channels, (self._global_params.num_classes - 1) * 2),
            )
        if self.task_mode in ['regress_rank_dorn', ]:
            self.regressioner_ = nn.Sequential(
                nn.Dropout(self._global_params.dropout_rate),
                nn.Linear(out_channels, (self._global_params.num_classes - 1) * 2),
            )
            self.ordinal_regression = OrdinalRegressionLayer()
        self._swish = MemoryEfficientSwish()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                # >>> import torch
                # >>> from efficientnet.model import EfficientNet
                # >>> inputs = torch.rand(1, 3, 224, 224)
                # >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                # >>> endpoints = model.extract_features(inputs)
                # >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                # >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                # >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                # >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                # >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f'reduction_{len(endpoints) + 1}'] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints[f'reduction_{len(endpoints) + 1}'] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        # x = self._dropout(x)
        # x = self._fc(x)
        # return x

        if self.task_mode == 'class':
            c_out = self.classifier_(x)
            return c_out
        elif self.task_mode == 'regress':
            r_out = self.regressioner_(x)
            return r_out[:, 0]
        elif self.task_mode == 'regress_rank_ordinal':
            r_out = self.regressioner_(x)
            r_out = r_out.view(-1, (self._global_params.num_classes - 1), 2)
            probas = F.softmax(r_out, dim=2)[:, :, 1]
            return r_out, probas
        elif self.task_mode in ['regress_rank_dorn', ]:
            r_out = self.regressioner_(x)
            predicts, softmax = self.ordinal_regression(r_out)
            return predicts, softmax
        elif self.task_mode == 'multi':
            c_out = self.classifier_(x)
            r_out = self.regressioner_(x)
            return c_out, r_out[:, 0]
        else:
            print(f'Do not support: {self.task_mode}'
                  f'Only support one of [multi, class, and regress] task_mode')

    @classmethod
    def from_name(cls, task_mode, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            task_mode (str): class, multi, regress
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(task_mode, blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, task_mode, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """create an efficientnet model according to name.

        Args:
            task_mode (str): class, multi, regress
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(task_mode, model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000),
                                advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]

        # Support the construction of 'efficientnet-l2' without pretrained weights
        valid_models += ['efficientnet-l2']

        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


def jl_efficientnet(task_mode='class', pretrained=True, num_classes=4, **kwargs):
    """
    Joint_learning efficient net

    Args:
        task_mode (string): multi, class, regress
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): number of class or number of output node
    """
    func = EfficientNet.from_pretrained if pretrained else EfficientNet.from_name
    model = func(task_mode=task_mode, model_name='efficientnet-b0', num_classes=num_classes)
    return model


# def _test():
#     net = jl_efficientnet(task_mode='REGRESS_rank_ordinal', pretrained=True, num_classes=4).cuda()
#     y_class, y_regres = net(torch.randn(48, 3, 224, 224).cuda())
#     print(y_class.size(), y_regres.size())
#     # y_class = net(torch.randn(48, 3, 224, 224).cuda())
#     # print(y_class.size())
#
#     # model = net.cuda()
#     # summary(model, (3, 224, 224))
# _test()

def label_to_levels(label, num_classes=4):
    levels = [1] * label + [0] * (num_classes - 1 - label)
    levels = torch.tensor(levels, dtype=torch.float32)
    return levels


def labels_to_labels(class_labels):
    """
    class_labels = [2, 1, 3]
    """
    levels = []
    for label in class_labels:
        levels_from_label = label_to_levels(int(label), num_classes=4)
        levels.append(levels_from_label)
    return torch.stack(levels).cuda()


def cost_fn(logits, label):
    num_classes = 4
    imp = torch.ones(num_classes - 1, dtype=torch.float).cuda()
    levels = labels_to_labels(label)
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1] * levels
                       + F.log_softmax(logits, dim=2)[:, :, 0] * (1 - levels)) * imp, dim=1))
    return torch.mean(val)


def loss_fn2(logits, label):
    num_classes = 4
    imp = torch.ones(num_classes - 1, dtype=torch.float)
    levels = labels_to_labels(label)
    val = (-torch.sum((F.logsigmoid(logits) * levels
                       + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                      dim=1))
    return torch.mean(val)
