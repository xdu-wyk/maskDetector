import torch
import math
import numpy as np
from PIL import Image, ImageDraw
import cv2
from torch.autograd import Variable
from get_nets import PNet, RNet, ONet, VggNet
from torchvision.transforms import transforms
from utils import (
    try_gpu,
    nms,
    calibrate_box,
    convert_to_square,
    correct_bboxes,
    get_image_boxes,
    generate_bboxes,
    preprocess,
)


class MaskDetector:
    def __init__(self, device=try_gpu()):
        self.device = device
        self.model = VggNet().to(device)
        self.model.eval()

    def detect(self, image):
        """
        利用VGG16进行口罩识别
        Args:
            image: tensor

        Returns:
            0 or 1，分别表示佩戴及未佩戴口罩
        """
        image = image.to(self.device)
        output = self.model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()


class FaceDetector:
    def __init__(self, device=try_gpu()):
        self.device = device

        # LOAD MODELS
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.onet.eval()
        self.maskDetector = MaskDetector()

    def detect(
            self,
            image,
            min_face_size=20.0,
            thresholds=[0.6, 0.7, 0.8],
            nms_thresholds=[0.7, 0.7, 0.7],
    ):
        """
        使用MTCNN人脸检测
        Arguments:
            image: PIL.Image.
            min_face_size: float.
            thresholds: list[3].
            nms_thresholds: list[3].

        Returns:
            [n_boxes, 5] and [n_boxes, 10],
            人脸框及五个校准点
        """

        # IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        scales = []

        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # 阶段 1

        bounding_boxes = []

        for s in scales:
            boxes = self.__run_first_stage(image, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # 阶段 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes).to(self.device))
            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # 阶段 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes).to(self.device))
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = (
                np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        )
        landmarks[:, 5:10] = (
                np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        )

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    def detectMask(self, image):
        """
        将佩戴及未佩戴口罩人脸进行颜色框区分
        Args:
            image: PIL.Image

        Returns:
            PIL.Image
        """
        size = 224
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        bounding_boxes, _ = self.detect(image)
        # convert bboxes to square
        square_bboxes = convert_to_square(bounding_boxes)
        for b in square_bboxes:
            face_img = image.crop((b[0], b[1], b[2], b[3]))
            face_img = face_img.resize((size, size), Image.BILINEAR)
            result = self.maskDetector.detect(transform(face_img).unsqueeze(0))
            color = "red" if result == 1 else "green"
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=color, width=3)
        return img_copy

    def __run_first_stage(self, image, scale, threshold):
        """使用PNet，选定人脸框，并使用NMS算法

        Arguments:
            image: PIL.Image.
            scale: float,
            threshold: float,

        Returns:
            float numpy array: shape [n_boxes, 9],
        """

        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, "float32")

        with torch.no_grad():
            img = Variable(torch.FloatTensor(preprocess(img)).to(self.device))
            output = self.pnet(img)
            probs = output[1].cpu().data.numpy()[0, 1, :, :]
            offsets = output[0].cpu().data.numpy()

        boxes = generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]
