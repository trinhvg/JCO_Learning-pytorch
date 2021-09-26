import imgaug  # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
import imgaug as ia


####
class Config(object):
    def __init__(self, _args=None):
        if _args is not None:
            self.__dict__.update(_args.__dict__)
        self.seed = 5 #self.seed
        self.infer_batch_size = 128
        self.nr_classes = 4

        # nr of processes for parallel processing input
        self.nr_procs_valid = 8

        self.load_network = False
        self.save_net_path = ""

        self.dataset = 'colon_manual'
        self.logging = False  # True for debug run only
        self.log_path = ""
        self.chkpts_prefix = 'model'
        self.model_name = 'validator'
        self.log_dir = self.log_path + self.model_name
        print(self.model_name)

    ####
    def infer_augmentors(self):
        if self.dataset == "prostate_hv":
            shape_augs = [
                iaa.Resize(0.5, interpolation='nearest'),
                iaa.CropToFixedSize(width=350, height=350, position="center"),
            ]
        else:
            shape_augs = []
        return shape_augs, None

############################################################################