import re
import json
from PIL import Image
import torch
import numpy as np
import os.path as osp
import torch.utils.data as data
from util.misc import NestedTensor
import dataset.transforms as T
from transformers import RobertaModel, RobertaTokenizerFast


##对图像进行数据增强和变换
def make_coco_transforms(image_set, cautious):
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [680, 712, 744, 776, 808, 840, 872, 904, 936, 968, 1000]

    max_size = 1200
    if image_set == "strain":
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize([600, 800, 1000]),
                            T.RandomSizeCrop(484, max_size, respect_boxes=cautious),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    else:
        return T.Compose(
            [
                T.RandomResize([max_size], max_size=max_size),
                normalize,
            ]
        )



class SKVGDataset(data.Dataset):
    def __init__(self, dataset_root, split, is_train=False):
        super(SKVGDataset, self).__init__()

        self.im_dir = osp.join(dataset_root, 'images')

        dataset_path = osp.join(dataset_root, 'annotations.json')
        dataset_path = osp.join(dataset_root, 'annotations_with_coref_label.json')
        # dataset_path = osp.join(dataset_root, 'split_anotations_cr.json')


        dataset = json.load(open(dataset_path, 'r'))
        self.data = dataset[split]
        self.split = split
        self._transforms = make_coco_transforms(split, True)
        self.is_train = is_train

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            './models/cache_dir/ssss')

    def __getitem__(self, idx):

        cur_example = self.data[idx]
        caption = cur_example['knowledge']
        refer = cur_example['ref_exp']


        img_file = cur_example['image_name']
        bbox_dict = cur_example['bbox']
        if self.split == 'train':
            cr_label = cur_example["n_k"]
            # cr_label = cur_example["coref_matrix"]


        # skvg的格式很奇葩，是左上角坐标加宽高
        bbox = [0] * 4
        bbox[0] = bbox_dict['x'] - bbox_dict['width'] / 2
        bbox[1] = bbox_dict['y'] - bbox_dict['height'] / 2
        bbox[2] = bbox_dict['width']
        bbox[3] = bbox_dict['height']

        bbox = np.array(bbox, dtype=int)
        bbox = torch.tensor(bbox)
        boxes = bbox.float()  # [4]

        img_path = osp.join(self.im_dir, img_file)
        try:
            img = Image.open(img_path).convert("RGB")  # PIL.Image类型的图片信息
        except:
            img_path = osp.join('./dataset/coco', img_file)
            img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # [n_obj,4]


        if self.split == 'train':
            cr_label = torch.as_tensor(cr_label, dtype=torch.float32)  # [n_obj,4]

        # 将box坐标从左上角坐标、宽，高 转换为 左上角坐标、右下角坐标
        boxes[:, 2:] += boxes[:, :2]
        # 限制坐标不要超出图像大小，超出就不合理了
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 将以上数据收集到 target中
        target = {}
        if 'new_ref' in cur_example:
            new_refer = cur_example['new_ref']
            target["new_ref"] = new_refer

        target["boxes"] = boxes
        if self.split == 'train':
            target['cr_label'] = cr_label

        target["img_path"] = img_path
        if refer is not None:
            target["ref_exp"] = refer
            s = refer.split()
        if caption is not None:
            target["caption"] = caption

        ## for conversion to coco api 添加以下数据来符合 coco api的格式

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        # 图片的变换和增强
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        _, h, w = img.shape
        target["size"] = torch.as_tensor([int(w), int(h)])


        return img, target

    def __len__(self):
        return len(self.data)

    ## 定义dataloader中的 collate_fn函数，其中的do_round一般默认为False,不会对图像的大小进行舍入
    def collate_fn(self, batch, do_round=False):
        # batch中的元素应该由两部分组成，一部分是[c，h,w]的原始图像信息，一部分是与该图像相关的数据字典
        # 将batch个元素拼接成一个有两个元素的列表，其中元素0中是图像信息，元素1中是图像相关数据
        bs = len(batch)
        batch = list(zip(*batch))
        final_batch = {}
        ## 将batch的两个元素赋值到final_batch中
        final_batch["samples"] = NestedTensor.from_tensor_list(batch[0],
                                                               do_round)  # （[bs,c,w,h]),[bs,w,h]）,c,w,h是一个维度中的最大值
        targets = batch[1]

        # text = ['reference: ' + t["ref_exp"] + 'knowledge: ' + t["caption"] for t in targets]
        captions = [t["caption"] for t in targets]
        refers = [t["ref_exp"] for t in targets]


        # tokenized_t = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt",
        #                                                return_special_tokens_mask=True)
        tokenized_c = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt",
                                                       return_special_tokens_mask=True)
        tokenized_r = self.tokenizer.batch_encode_plus(refers, padding="longest", return_tensors="pt",return_special_tokens_mask = True)



        # text_input_ids = tokenized_t.input_ids
        # text_attention_mask = tokenized_t.attention_mask

        captions_input_ids = tokenized_c.input_ids
        refer_input_ids = tokenized_r.input_ids

        captions_attention_mask = tokenized_c.attention_mask
        refer_attention_mask = tokenized_r.attention_mask




        if self.split == 'train':
            cr_label_list = []
            ref_len = refer_input_ids.size(1)
            cap_len = captions_input_ids.size(1)

            for cr_label in [t["cr_label"] for t in targets]:
                rlen, clen = cr_label.size()
                cur_cr_label = torch.zeros(ref_len, cap_len)
                cur_cr_label[:min(rlen, ref_len), :min(clen, cap_len)] = cr_label[:min(rlen, ref_len),
                                                                         :min(clen, cap_len)]
                cr_label_list.append(cur_cr_label)

            final_batch["cr_labels"] = torch.stack(cr_label_list, dim=0)  # [bs, ref_len, cap_len]
        else:
            final_batch["cr_labels"] = torch.zeros(bs, refer_input_ids.size(1), captions_input_ids.size(1))


        # final_batch["texts"] = NestedTensor(text_input_ids, text_attention_mask)
        final_batch["captions"] = NestedTensor(captions_input_ids, captions_attention_mask)
        final_batch["refers"] = NestedTensor(refer_input_ids, refer_attention_mask)

        final_batch["targets"] = batch[1]  # 一个列表，所以不用考虑不同样本数据维度不同的问题

        return final_batch