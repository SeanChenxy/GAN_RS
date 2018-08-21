from .base_options import BaseOptions

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--show_video', action='store_true', help='Whether show fake video or not')
        self.parser.add_argument('--writename', type=str, default=None, help='The name to write video')
        self.isTrain = False
        ### Detection ###
        self.parser.add_argument('--model_dir', type=str, default=None, help='Detection .pth file')
        self.parser.add_argument('--backbone', type=str, default=None, help='Detection model')
        self.parser.add_argument('--ssd_dim', type=int, default=320, help='SSD detection size ')
        self.parser.add_argument('--bn', type=str2bool, default=False, help='model option')
        self.parser.add_argument('--refine', type=str2bool, default=False, help='model option')
        self.parser.add_argument('--deform', type=int, default=0, help='model option ')
        self.parser.add_argument('--multihead', type=str2bool, default=False, help='model option')
        self.parser.add_argument('--tub', type=int, default=0, help='OTA option ')
        self.parser.add_argument('--tub_thresh', type=float, default=1., help='OTA option ')
        self.parser.add_argument('--tub_generate_score', type=float, default=0.1, help='OTA option ')
        self.parser.add_argument('--confidence_threshold', type=float, default=0.4, help='NMS option ')
        self.parser.add_argument('--nms_threshold', type=float, default=0.3, help='NMS option ')
        self.parser.add_argument('--top_k', type=int, default=400, help='NMS option ')











