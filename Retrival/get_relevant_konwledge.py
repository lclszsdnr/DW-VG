import json
import tqdm
from dataset.data_loader_old import SKVGDataset ,make_coco_transforms
import torch
from PIL import Image
from util.box_ops import box_cxcywh_to_xyxy,box_iou
from torch.utils.data import DataLoader
from models.backbone import Backbone, Joiner, TimmBackbone
from models.coref_deter import MDETR
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
import matplotlib
# matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt
import cv2
import numpy as np
import  seaborn as sns
import torch.nn.functional as F


def compute_sentence_importance(importance_scores, s, e, window_size):
    """
    对 [s, e) 范围内的 token 使用指定窗口大小进行滑动，
    返回所有窗口平均值中的最大值作为该句子的“重要度”。
    如果句子长度小于窗口大小，则窗口大小取句子长度。
    """
    length = e - s
    if length == 0:
        return 0.0
    if length < window_size:
        window_size = length
    max_avg = 0.0
    # 滑动窗口：从 s 到 e - window_size
    for i in range(s, e - window_size + 1):
        window_avg = importance_scores[i:i + window_size].mean().item()
        if window_avg > max_avg:
            max_avg = window_avg
    return max_avg


def extend_token_span(importance_scores, start, end, threshold):
    """正常扩展：在 [start, end) 基础上向两侧扩展，只要相邻 token 的重要度 ≥ threshold。"""
    while start > 0 and (importance_scores[start - 1] >= threshold  or importance_scores[start - 2] >= threshold):
        start -= 1
    # 向右扩展
    while (end < len(importance_scores) and importance_scores[end] >= threshold) or (end < len(importance_scores)-1 and importance_scores[end+1] >= threshold):
        end += 1
    return start, end

def trim_token_span(importance_scores, start, end, threshold):
    """
    削减边缘：从左侧向右找到第一个 token 重要度 ≥ threshold，
    从右侧向左找到最后一个 token 重要度 ≥ threshold，
    返回内部连续满足条件的 span，防止越界。
    """
    # 先向右移动 start，直到找到第一个 >= threshold 的 token
    while start < end  and start  <  len(importance_scores) and importance_scores[start] < threshold:
        start += 1

    # 再向左移动 end，直到找到最后一个 >= threshold 的 token
    while end > start and end <  len(importance_scores) and importance_scores[end - 1] < threshold:
        end -= 1

    # 确保返回的范围不会越界
    return start, end



def get_expanded_binary_mask(sentence_indices, importance_scores, threshold=0.5, threshold_extend=0.6):
    """
    根据新的句子重要度计算方式和 token 重要度，
    生成扩展后的 token 级别 binary mask。

    逻辑：
      1. 先以所有句子的最短长度作为滑动窗口大小，
         对每个句子计算滑动窗口平均值的最大值，作为句子的重要度。
      2. 选择主句：平均重要度最高的句子。
      3. 如果主句平均重要度 < threshold：
             使用 trim_token_span：仅保留主句中连续满足 token 重要度 ≥ threshold 的部分，
             并对该区域向左右扩展 token（仅扩展相邻 token 重要度 ≥ threshold）。
         否则（主句平均重要度 ≥ threshold）：
             检查主句左右相邻句子是否满足（按滑动窗口计算的）重要度 ≥ threshold，
             如果满足，则纳入。将所有被纳入句子的范围合并，
             对合并后的 span进行 token 级别扩展。
      4. 返回 binary mask（长度与 importance_scores 相同，选中 token 为 1，否则为 0）。
    """
    # 计算所有句子的长度，取最小值作为窗口大小
    window_size = min(e - s for s, e in sentence_indices if e - s > 0)
    if window_size<4:
        window_size=4

    # 使用滑动窗口计算每个句子的重要度
    sentence_importances = []
    for s, e in sentence_indices:
        imp = compute_sentence_importance(importance_scores, s, e, window_size)
        sentence_importances.append(imp)

    # 选择主句：重要度最高的句子
    most_important_idx = torch.tensor(sentence_importances).argmax().item()

    # 初始化 mask 全零
    mask = torch.zeros_like(importance_scores, dtype=torch.float)

    # 情况 1：主句重要度 < threshold
    # 情况 2：主句重要度 ≥ threshold
    main_s, main_e = sentence_indices[most_important_idx]
    sentence_avg_scores = [
        importance_scores[s:e].mean().item() for s, e in sentence_indices
    ]
    main_importance = sentence_avg_scores[most_important_idx]

    if main_importance < threshold:
        # 主句较不重要，先“削减”边缘，仅保留连续满足 token 重要度 ≥ threshold 的部分，
        # 然后对该区域进行 token 级别扩展
        trimmed_s, trimmed_e = trim_token_span(importance_scores, main_s, main_e, threshold)
        final_main_s, final_main_e = extend_token_span(importance_scores, trimmed_s, trimmed_e, threshold_extend)
        main_span = (final_main_s, final_main_e)
    else:
        # 主句较重要，先将主句纳入
        selected_starts = [main_s]
        selected_ends = [main_e]
        # 检查左右相邻句子：仅当其滑动窗口计算的重要度 ≥ threshold 时纳入
        for offset in [-1, 1]:
            idx = most_important_idx + offset
            if 0 <= idx < len(sentence_indices):
                if sentence_importances[idx] >= threshold_extend:
                    s, e = sentence_indices[idx]
                    selected_starts.append(s)
                    selected_ends.append(e)
        aggregated_start = min(selected_starts)
        aggregated_end = max(selected_ends)

        # 对合并后的 span 进行 token级扩展
        final_main_s, final_main_e = extend_token_span(importance_scores, aggregated_start, aggregated_end, threshold_extend)
        final_main_s, final_main_e  = trim_token_span(importance_scores,final_main_s, final_main_e , threshold)

        main_span = (final_main_s, final_main_e)

    # 标记最终主句（或合并后主句）的区域
    mask[main_span[0]:main_span[1]] = 1.0

    return mask ,main_span



def _make_backbone(backbone_name: str, mask: bool = False):
    if backbone_name[: len("timm_")] == "timm_":
        backbone = TimmBackbone(
            backbone_name[len("timm_") :],
            mask,
            main_layer=-1,
            group_norm=True,
        )
    else:
        backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=False)

    hidden_dim = 256
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    return backbone_with_pos_enc

def _make_detr(
    backbone_name: str,
    num_queries=100,
    mask=False,

    text_encoder="roberta-base",
):
    hidden_dim = 256
    backbone = _make_backbone(backbone_name, mask)
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True, text_encoder_type=text_encoder)
    detr = MDETR(
        backbone,
        transformer,
        num_queries=num_queries,
    )

    return detr



def bbox_to_rect(bbox, color,ration,wh,title=None ):

    bbox[0], bbox[2] =  bbox[0]*wh[0], bbox[2] *wh[0]
    bbox[1], bbox[3] = bbox[1]*wh[1], bbox[3] *wh[1]


    bbox[0] = bbox[0]-bbox[2]/2
    bbox[1] = bbox[1] -bbox[3]/2
    bbox[2] = bbox[0]+bbox[2]
    bbox[3] = bbox[1]+bbox[3]


    bbox[0], bbox[2] =  bbox[0]/ration[0], bbox[2] /ration[0]
    bbox[1], bbox[3] = bbox[1]/ration[1], bbox[3] /ration[1]




    return bbox


def tshow(t_attention, x=False,y=False
          ,titel='text attention'):
    plt.figure(figsize=(10, 20))  # 这样才会正确设置大小
    sns.heatmap(t_attention.cpu().detach(),
                cbar=True,
                cmap="RdBu_r",
                annot=False,
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=y,
                xticklabels=x
                )
    plt.title(titel, fontsize=20)
    plt.tight_layout()
    plt.show()

def show(path,wh,pr,cap,v_attention,t_attention,vis_shape,i):
    pr =pr.tolist()
    img = Image.open(path)
    img_size = img.size
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(wh, img.size))
    bbox = bbox_to_rect(pr, 'red', ratios, wh)
    img_ori=cv2.imread(path)

    h,w = vis_shape
    v_attention = v_attention.reshape(h,w)

    layer_one = v_attention.cpu().detach().numpy()


    # tshow(t_attention.unsqueeze(0),x=cap,titel=cap[i])

    ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
    ret = ret.astype(np.uint8)
    gray = ret[:, :, None]
    ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    ret = cv2.resize(ret,img_size)

    show = ret * 0.50 + img_ori * 0.50
    cv2.rectangle(show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255,0,0), thickness=5)
    cv2.namedWindow(cap[i],cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cap[i],800,600)
    cv2.imshow(cap[i],show.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

split = 'test'
transformer = make_coco_transforms('val',False)
dataset = SKVGDataset(dataset_root='./dataset/sk_vg',split=split,)
dataloader = DataLoader(dataset,batch_size=1,collate_fn=dataset.collate_fn,shuffle=False)
tokenzier = dataset.tokenizer
model = _make_detr("resnet101").to(torch.device('cuda:0'))
# checkpoint = torch.load('./outputs/01/BEST_checkpoint.pth',map_location='cpu')

# checkpoint = torch.load('/media/team/data/CODE/slef_code/final_code/mdetr/outputs/o_wconf_wtoolcf/BEST_checkpoint.pth',map_location='cpu')

# checkpoint = torch.load('./models/cache_dir/pretrained_resnet101_checkpoint.pth',map_location='cpu')
# for k , v in checkpoint['model'].items():
#     # if 'transformer.encoder' in k or'text_encoder' in k:
#     if 'transformer.encoder' in k :
#
#         checkpoint['model'][k] = checkpoint0['model'][k]

# missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'],strict=False)
# torch.save(model.transformer.text_encoder.state_dict(),'oroberta_trained.pth')
# print(missing_keys)
model.eval()

new_refers = []
relevant_knowledge_map=[]
for l, final_batch in  enumerate(tqdm.tqdm(dataloader)):

    # if l <93:
    #     continue
    device = torch.device('cuda:0')
    img_data = final_batch["samples"].to(device)
    refer_data = final_batch['refers'].to(device)
    knowl_data = final_batch['captions'].to(device)
    # name_data = final_batch['names'].to(device)
    # mask_data = final_batch['names_mask'].to(device)
    batch_sentences_se = final_batch["batch_sentences_se"]

    targets = final_batch["targets"]

    memory_cache = model(img_data, refer_data, knowl_data,
                         encode_and_save=True)
    # 解码部分，获取最终解码输出，
    outputs = model(img_data, refer_data, knowl_data,
                    encode_and_save=False, memory_cache=memory_cache)


    len_knowl =  knowl_data.tensors.shape[1]
    len_refer = refer_data.tensors.shape[1]
    rk_att = memory_cache['t_logits']
    importance_threshold = 1.85
    extended_threshold = 2.0

    # 挨个 样本 进行分析
    for j in range(len(targets)):
        gt = targets[j]["boxes"]
        # pr = outputs["pred_boxes_q"][j]
        text_tensors = torch.cat([refer_data.tensors, knowl_data.tensors], dim=1)
        k_cap = tokenzier.convert_ids_to_tokens(knowl_data.tensors[j])
        r_cap = tokenzier.convert_ids_to_tokens(refer_data.tensors[j])
        t_cap = r_cap+k_cap

        ## 以当前 模型 输出 的 指代表达和  场景知识的 相似度 或者 是 注意力 作为 分数
        t_attention = rk_att[j]
        importance_scores = t_attention[1:len_refer-1,len_refer:-2].mean(dim=0) #获取 整个 指代表达 与 每个 知识的 注意力得分

        #获取 代表重要知识的 mask ， 和起始位置
        binary_mask,se_index_all = get_expanded_binary_mask(batch_sentences_se[j],importance_scores,importance_threshold,extended_threshold)

        is_include = True
        importance_scores= t_attention[1:4, len_refer:].mean(dim=0)
        subject_binary_mask, se_index_subject = get_expanded_binary_mask(batch_sentences_se[j], importance_scores, importance_threshold,extended_threshold)
        if binary_mask.sum() <4 or subject_binary_mask.sum() <4:
            is_include = False
            try:
                importance_scores,_ = t_attention[len_refer-4:len_refer-1, len_refer:].max(dim=0)

            except:
                importance_scores,_ = t_attention[len_refer-3:len_refer-1, len_refer:].max(dim=0)

            binary_mask, se_index= get_expanded_binary_mask(batch_sentences_se[j],importance_scores,importance_threshold,extended_threshold)
        if is_include:
            se_index = [min(se_index_all[0],se_index_subject[0]), min(se_index_all[1], se_index_subject[1])]
        token_ids =  knowl_data.tensors[:,se_index[0]:se_index[1]]
        decoded_text = tokenzier.decode(token_ids[0])

        if is_include:
            new_refer = targets[0]['ref_exp']+' '+":" + decoded_text.strip('<s> ')
        else:
            new_refer = targets[0]['ref_exp'] + ' ' + "," + decoded_text.strip('<s> ')

        relevant_knowledge_map.append(binary_mask.tolist())
        new_refers.append(new_refer)
        # tshow(binary_mask.unsqueeze(0),x=k_cap,y=r_cap[0],titel=r_cap)
        # tshow(t_attention[:len_refer,len_refer:], x=k_cap, y=r_cap[0], titel=r_cap)
json.dump(new_refers,open('o_new_refers_{}.json'.format(split),'w'))
json.dump(relevant_knowledge_map,open('o_relevant_knowledge_map_{}.json'.format(split),'w'))




