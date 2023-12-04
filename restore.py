import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper

from basicsr.utils.registry import ARCH_REGISTRY


class Restore:
    def __init__(self):
        self.upscale = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bg_upsampler = self.set_realesrgan()
        self.detection_model = 'retinaface_resnet50'
        self.only_center_face = False
        self.fidelity_weight = 0.5
        self.draw_box = False
        self.bg_tile = 400
        self.pretrain_model_url = {
            'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        }

    def set_realesrgan(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer

        use_half = False
        if torch.cuda.is_available():  # set False in CPU/MPS mode
            no_half_gpu_list = ['1650', '1660']  # set False for GPUs that don't support f16
            if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
                use_half = True

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
            model=model,
            tile=self.bg_tile,
            tile_pad=40,
            pre_pad=0,
            half=use_half
        )

        return upsampler

    def do(self, output_path, image_path):

        # ------------------ set up CodeFormer restorer -------------------
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                              connect_list=['32', '64', '128', '256']).to(self.device)

        # ckpt_path = 'weights/CodeFormer/codeformer.pth'
        ckpt_path = load_file_from_url(url=self.pretrain_model_url['restoration'],
                                       model_dir='~/.cache/CodeFormer/weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()

        # ------------------ set up FaceRestoreHelper -------------------
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        if self.bg_upsampler is not None:
            print(f'Background upsampling: True, Face upsampling: RealESRGAN')

        face_helper = FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=self.detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device)

        # -------------------- start to processing ---------------------
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        face_helper.read_image(img)
        # get face landmarks for each face
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=self.only_center_face, resize=640, eye_dist_threshold=5)
        print(f'\tdetect {num_det_faces} faces')
        # align and warp each face
        face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=self.fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        # upsample the background
        if self.bg_upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box)

        # save restored img
        if restored_img is not None:
            imwrite(restored_img, output_path)
