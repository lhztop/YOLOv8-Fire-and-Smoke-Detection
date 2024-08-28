#coding: utf8
import logging
import os.path
import random
import ultralytics
from PIL import Image
from typing import Literal

from imagededup.methods import PHash
import torch


class YOLOFormatMerge(object):
    input_dir:str = 'datasets/fire_smoke_data_set/fireandsmoke'
    output_dir:str = 'datasets/merge'
    YOLO_CLASSES = {0: 'Fire', 1: "defaults", 2: "smoke"}
    merge_main_image_dir = 'datasets/fire-8/train/images'
    label_dir = "yolo"
    image_dir = "images"

    def __init__(self, input_dir:str = 'datasets/fire_smoke_data_set/fireandsmoke', output_dir:str = 'datasets/merge'):
        import os
        if input_dir is not None:
            self.input_dir = input_dir
        if output_dir is not None:
            self.output_dir = output_dir
        if not self.input_dir.startswith("/"):
            filename =  os.path.abspath(__file__)
            self.root_dir = os.path.dirname(filename)
            self.input_dir = os.path.join(self.root_dir, self.input_dir)
            self.output_dir = os.path.join(self.root_dir, self.output_dir)
            self.merge_main_image_dir = os.path.join(self.root_dir, self.merge_main_image_dir)
        else:
            self.root_dir = None

    def can_add_to_merge(self, image_file_name, process_image_files:set, main_vectors, dup_files):
        if image_file_name in dup_files:
            if dup_files[image_file_name] is not None and len(dup_files[image_file_name]) > 0:  # dup images

                for dup_file in dup_files[image_file_name]:
                    if dup_file in main_vectors:
                        process_image_files.add(image_file_name)
                        return False
                    if dup_file in process_image_files:
                        return False
        process_image_files.add(image_file_name)
        return True

    def convert(self):
        import glob
        files = glob.glob(f'{self.input_dir}/{self.label_dir}/*.txt')
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels", exist_ok=True)
        import shutil
        dedup_obj = PHash(verbose=True)
        main_vectors = dedup_obj.encode_images(os.path.join(self.output_dir, 'images'))
        main_vectors = {os.path.join(self.output_dir, 'images', k): v for k, v in main_vectors.items()}
        merge_vectors = dedup_obj.encode_images(os.path.join(self.input_dir, self.image_dir))
        merge_vectors = {os.path.join(self.input_dir, self.image_dir, k): v for k, v in merge_vectors.items()}
        merge_vectors.update(main_vectors)
        dup_files = dedup_obj.find_duplicates(encoding_map=merge_vectors, max_distance_threshold=6)

        processed_image_files = set()
        for f in files:
            filename = os.path.basename(f)
            image_name = filename.replace(".txt", "")
            image_file_name = f'{self.input_dir}/images/{image_name}.jpg'
            image_file_basename = os.path.basename(image_file_name)
            if not self.can_add_to_merge(image_file_name, processed_image_files, main_vectors, dup_files):
                logging.error(f"{image_file_name} duplicate with {dup_files[image_file_name]}")
                continue
            with open(f) as fp:
                lines = fp.readlines()
                if lines is None or len(lines) <= 0:
                    logging.error(f"{f} cotains no annotation")
                    continue
                with open(f"{self.output_dir}/labels/{filename}", 'w') as labelfp:
                    for line in lines:
                        if line == "":
                            continue
                        strs = line.split(" ")
                        raw_class = int(strs[0])
                        yolo_class = self.transfer_yolo_classes(raw_class)
                        strs[0] = str(yolo_class)
                        line = " ".join(strs)
                        labelfp.write(line)
                shutil.copy(image_file_name, f"{self.output_dir}/images")

    def transfer_yolo_classes(self, raw_class:int):
        if raw_class == 0:
            return 0
        elif raw_class == 1:
            return 2
        else:
            logging.error(f"{raw_class} not support in YOLOFormatMerge")
            return None


class YOLOImageLableFormatMerge(YOLOFormatMerge):
    label_dir = "labels"
    image_dir = "images"
    input_dir = "datasets/fire-8/train"

    def transfer_yolo_classes(self, raw_class:int):
        return raw_class

    def __init__(self, input_dir: str = 'datasets/fire-8/train', output_dir: str = 'datasets/merge'):
        super().__init__(input_dir, output_dir)



class VOCFormatMerge(YOLOFormatMerge):
    image_dir = "JPEGImages"
    annotation_dir = "Annotations"
    def __init__(self, input_dir:str = 'datasets/fire_smoke_data_set/火灾/fireDetectVOCfinal', output_dir:str = 'datasets/merge'):
        super().__init__(input_dir, output_dir)

    def transfer_yolo_classes(self, raw_class:str):
        if raw_class == "smoke":
            return 2
        elif raw_class == "fire":
            return 0
        else:
            return 1

    def convert(self):
        import glob
        files = glob.glob(f'{self.input_dir}/JPEGImages/*.*')
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels", exist_ok=True)
        import shutil
        dedup_obj = PHash(verbose=True)
        main_vectors = dedup_obj.encode_images(os.path.join(self.output_dir, 'images'))
        main_vectors = {os.path.join(self.output_dir, 'images', k):v for k, v in main_vectors.items()}
        merge_vectors = dedup_obj.encode_images(os.path.join(self.input_dir, self.image_dir))
        merge_vectors = {os.path.join(self.input_dir, self.image_dir, k): v for k, v in merge_vectors.items()}
        merge_vectors.update(main_vectors)
        dup_files = dedup_obj.find_duplicates(encoding_map=merge_vectors, max_distance_threshold=8)
        import pathlib
        import lxml.etree
        processed_image_files = set()
        for image_file_name in files:
            xml_name = pathlib.Path(image_file_name).stem
            xml_file_name = f'{self.input_dir}/{self.annotation_dir}/{xml_name}.xml'
            if not self.can_add_to_merge(image_file_name, processed_image_files, main_vectors, dup_files):
                logging.error(f"{image_file_name} duplicate with {dup_files[image_file_name]}")
                continue
            with open(xml_file_name) as fp:
                tree = lxml.etree.parse(fp)
                width = int(tree.xpath('/annotation/size/width')[0].text)
                height = int(tree.xpath('/annotation/size/height')[0].text)
                lines = tree.xpath('/annotation/object')
                if lines is None or len(lines) <= 0:
                    logging.error("{xml_file_name} no objects")
                    continue
                with open(f"{self.output_dir}/labels/{xml_name}.txt", 'w') as labelfp:
                    for line in lines:
                        raw_class = line.xpath('name')[0].text
                        x1 = float(line.xpath('bndbox/xmin')[0].text)
                        x2 = float(line.xpath('bndbox/xmax')[0].text)
                        y1 = float(line.xpath('bndbox/ymin')[0].text)
                        y2 = float(line.xpath('bndbox/ymax')[0].text)

                        strs = ["", "", "", "", ""]
                        yolo_class = self.transfer_yolo_classes(raw_class)
                        strs[0] = str(yolo_class)
                        strs[1] = str(x1/width)
                        strs[2] = str(y1/height)
                        strs[3] = str((x2-x1)/width)
                        strs[4] = str((y2-y1)/height)
                        line = " ".join(strs) + "\n"
                        labelfp.write(line)
                shutil.copy(image_file_name, f"{self.output_dir}/images")


class YOLO2VitFormatConverter(YOLOFormatMerge):

    input_dir:str = 'datasets/merge'
    output_dir:str = 'datasets/vit'
    label_dir = "labels"
    image_dir = "images"
    YOLO_CLASSES = {0: 'Fire', 1: "defaults", 2: "smoke"}

    def __init__(self, input_dir:str = 'datasets/merge', output_dir:str = 'datasets/vit'):
        super().__init__(input_dir, output_dir)

    def convert(self):
        import glob
        files = glob.glob(f'{self.input_dir}/{self.image_dir}/*.*')
        os.makedirs(f"{self.output_dir}/smoke", exist_ok=True)
        os.makedirs(f"{self.output_dir}/defaults", exist_ok=True)
        os.makedirs(f"{self.output_dir}/Fire", exist_ok=True)
        import shutil
        dedup_obj = PHash(verbose=True)
        main_vectors = dedup_obj.encode_images(self.output_dir, recursive=True)
        main_vectors = {os.path.join(self.output_dir, k): v for k, v in main_vectors.items()}
        merge_vectors = dedup_obj.encode_images(os.path.join(self.input_dir, self.image_dir))
        merge_vectors = {os.path.join(self.input_dir, self.image_dir, k): v for k, v in merge_vectors.items()}
        merge_vectors.update(main_vectors)
        dup_files = dedup_obj.find_duplicates(encoding_map=merge_vectors, max_distance_threshold=8)

        processed_image_files = set()
        import pathlib
        for image_file_name in files:
            label_file_name = pathlib.Path(image_file_name).stem
            xml_file_name = f'{self.input_dir}/{self.label_dir}/{label_file_name}.txt'
            if not self.can_add_to_merge(image_file_name, processed_image_files, main_vectors, dup_files):
                logging.error(f"{image_file_name} duplicate with {dup_files[image_file_name]}")
                continue
            if not self.can_add_to_merge(image_file_name, processed_image_files, main_vectors, dup_files):
                logging.error(f"{image_file_name} duplicate with {dup_files[image_file_name]}")
                continue
            with open(xml_file_name) as fp:
                lines = fp.readlines()
                classes = set()
                if lines is None or len(lines) <= 0:
                    logging.error(f"{xml_file_name} cotains no annotation")
                    continue
                for line in lines:
                    if line == "":
                        continue
                    strs = line.split(" ")
                    raw_class = int(strs[0])
                    classes.add(raw_class)
                if 0 in classes:
                    shutil.copy(image_file_name, f"{self.output_dir}/{self.YOLO_CLASSES[0]}")
                elif 2 in classes:
                    shutil.copy(image_file_name, f"{self.output_dir}/{self.YOLO_CLASSES[2]}")
                elif 1 in classes:
                    shutil.copy(image_file_name, f"{self.output_dir}/{self.YOLO_CLASSES[1]}")
                else:
                    logging.error(f"{classes} not yolo class")


class Vit2VitFormatConverter(YOLO2VitFormatConverter):
    input_dir: str = 'datasets/merge'
    output_dir: str = 'datasets/vit'

    VIT_CLASS_MAP = {"fire": 'Fire', "": "defaults", "smoke": "smoke"}

    def __init__(self, input_dir: str = 'datasets/merge', output_dir: str = 'datasets/vit'):
        super().__init__(input_dir, output_dir)

    def convert(self):
        pass



class SmokeFireBenchmark(object):
    _model = None
    _score_threshold = 0.45
    fp:int = 0  # 预测为正，实际为负
    fn:int = 0  # 预测为负，实际为正
    tp:int = 0  # 预测为正，实际为正
    tn:int = 0  # 预测为负，实际为负
    total_true:int = 0
    total_false:int = 0


    def __init__(self, model:str = "train/best.pt", score_threshold: float = 0.45, model_type: Literal["yolo", "google-vit"] = "yolo"):
        self._score_threshold = score_threshold
        if model_type == "yolo":
            from ultralytics import YOLO
            self._model = YOLO(model)
        else:
            from transformers import ViTImageProcessor, ViTForImageClassification
            image_processor = ViTImageProcessor.from_pretrained(model)
            model = ViTForImageClassification.from_pretrained(model)
            self._model = (image_processor, model)

    def get_yolo_label(self, val_dir:str=None):
        import glob
        files = glob.glob(f'{val_dir}/**/*.txt', recursive=True)
        labels = dict()
        for f in files:
            with open(f) as fp:
                lines = fp.readlines()
                classes = set()
                if lines is None or len(lines) <= 0:
                    logging.error(f"{f} cotains no annotation")
                    continue
                for line in lines:
                    if line == "":
                        continue
                    strs = line.split(" ")
                    raw_class = int(strs[0])
                    classes.add(raw_class)
                labels[f] = classes
        return labels

    def get_yolo_predict(self, image_file_name:str):
        pred = self._model.predict(image_file_name)
        if not pred or len(pred) == 0:
            return None
        ret = set()
        for res in pred:
            detection_count = res.boxes.shape[0]
            for i in range(detection_count):
                cls = int(res.boxes.cls[i].item())
                name = res.names[cls]
                confidence = float(res.boxes.conf[i].item())
                if confidence >= self._score_threshold:
                    ret.add(cls)
        if len(ret) == 0:
            ret.add(1)
        return ret

    def get_vit_predict(self, image_file_name:str):
        image = Image.open(image_file_name)
        if image_file_name.lower().endswith(".png"):
            image = image.convert("RGB")
        image_processor, model = self._model
        inputs = image_processor(images=image, return_tensors="pt")

        inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
        outputs = model(**inputs)
        res = torch.topk(outputs.logits.softmax(dim=-1), 5)
        confidence = [x.item() for x in res[0][0, :]]
        class_ids = [x.item() for x in res[1][0, :]]
        class_list = [model.config.id2label[x] for x in class_ids]

        # ret = {'class_id_list': class_ids, 'score_list': confidence, 'label_list': class_list}
        ret = set()
        for i in range(len(class_ids)):
            if confidence[i] >= self._score_threshold:
                ret.add(class_ids[i])
        if len(ret) <= 0:
            ret.add(1)
        return ret

    def calc_single_pre_true_score(self, pre_result, true_result, merge_smoke_fire_2_fire:bool=True):
        if 1 in true_result and len(true_result) > 1:
            logging.error("default contains smoke or fire in true")
            true_result.remove(1)

        if 1 in pre_result and len(pre_result) > 1:
            logging.error("default contains smoke or fire in predict")
            pre_result.remove(1)
        if merge_smoke_fire_2_fire:
            if 0 in true_result and 2 in true_result:
                true_result.remove(2)
            if 0 in pre_result and 2 in pre_result:
                pre_result.remove(2)

        for clss in true_result:
            if clss == 1:
                self.total_false += 1
            else:
                self.total_true += 1
        for pre_clss in pre_result:
            if pre_clss == 1:
                if pre_clss in true_result:
                    self.tn += 1
                else:
                    self.fn += 1
            else:
                if pre_clss in true_result:
                    self.tp += 1
                else:
                    self.fp += 1


    def calc(self, val_dir:str="datasets/output/val"):
        import glob
        files = glob.glob(f'{val_dir}/**/images/*.*', recursive=True)
        import pathlib
        labels = dict()
        yolo_val = self.get_yolo_label(val_dir)
        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.tn = 0
        self.total_true = 0
        self.total_false = 0
        for f in files:
            ext = pathlib.Path(f).suffix
            label_file_name = f.replace('/images/', '/labels/')[:-len(ext)] + ".txt"
            if isinstance(self._model, tuple):
                pred_result = self.get_vit_predict(f)
            else:
                pred_result = self.get_yolo_predict(f)
            true_result = yolo_val[label_file_name]
            self.calc_single_pre_true_score(pred_result, true_result)
        print(f" tp+tn should = total true, {self.tp+self.fn} = {self.total_true}, {self.tp + self.fn == self.total_true }")
        header = ["Accuracy", "Precision", "Recall", "F1-score"]
        value = [(self.tp + self.tn) / (self.total_true + self.total_false), self.tp/(self.tp + self.fp), self.tp/self.total_true]
        value.append(2*value[1]*value[2]/(value[1] + value[2]))  # f1 = 2p*r/p+r = 2/(1/p+1/r)
        row_format = "{:>15}" * len(header)
        print(row_format.format(*header))
        print(row_format.format(*value))


class YOLOSampler(YOLOFormatMerge):
    input_dir: str = 'datasets/merge'
    output_dir: str = 'datasets/output'

    ratio = {"train":7, "val": 2, "test":1}
    label_dir = "labels"
    image_dir = "images"

    def __init__(self, input_dir:str = 'datasets/merge', output_dir:str = 'datasets/output'):
        super().__init__(input_dir, output_dir)

    def sample(self):
        import glob
        files:list = glob.glob(f'{self.input_dir}/images/*.*')
        for dir in self.ratio:
            os.makedirs(f"{self.output_dir}/{dir}", exist_ok=True)
        random.shuffle(files)
        total = 0
        for k, v in self.ratio.items():
            total += v
        train_count = int(self.ratio["train"] / total * len(files))
        val_count = int(self.ratio["val"] / total * len(files))
        test_count = int(self.ratio["train"] / total * len(files))
        counts = {"train": train_count, "val": val_count, "test": test_count}

        start_idx = 0
        import shutil, pathlib
        for dir in ["train", "val", "test"]:
            end_idx = start_idx + counts[dir]
            if "test" == dir:
                end_idx = -1
            sample_files = files[start_idx: end_idx]
            os.makedirs(f"{self.output_dir}/{dir}/images", exist_ok=True)
            os.makedirs(f"{self.output_dir}/{dir}/labels", exist_ok=True)
            for f in sample_files:
                label_file_name_extension = pathlib.Path(f).suffix
                shutil.copy(f, f"{self.output_dir}/{dir}/images")
                shutil.copy((f[:-len(label_file_name_extension)] + ".txt").replace(f"/{self.image_dir}/", f"/{self.label_dir}/"), f"{self.output_dir}/{dir}/labels")
            start_idx = end_idx


if __name__ == "__main__":
    # merger = YOLOFormatMerge()
    # merger.convert()
    # merger = YOLOImageLableFormatMerge()
    # merger.convert()
    # merger = VOCFormatMerge()
    # merger.convert()
    # merger = VOCFormatMerge(input_dir="datasets/fire_smoke_data_set/bd_fire/VOC2020")
    # merger.convert()
    # merge = VOCFormatMerge(input_dir="datasets/fire_smoke_data_set/bd_fire/fire_smoke")
    # merge.image_dir = "images"
    # merge.annotation_dir = "annotations"
    # merge.convert()
    #
    # converter = YOLO2VitFormatConverter()
    # converter.convert()
    # sampler = YOLOSampler()
    # sampler.sample()
    import sys
    if len(sys.argv) < 3:
        print("Usage: benchark model_name model_type\n, model_type: yolo|google-vit\n")
        sys.exit(1)
    benchmark = SmokeFireBenchmark(model=sys.argv[1], model_type=sys.argv[2])
    benchmark.calc()