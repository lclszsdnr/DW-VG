from maverick import Maverick
import json
from transformers import RobertaTokenizerFast,BertTokenizerFast
from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast

import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch


def draw_coref_matrix_vertical_tokens(matrix, tokens, cell_size=30, font_size=10):
    num_tokens = len(tokens)
    image_width = cell_size * (num_tokens + 1)
    image_height = cell_size * (num_tokens + 1)
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for idx, token in enumerate(tokens):
        x = (idx + 1) * cell_size + 5
        y = cell_size - 2
        for ch in token:
            draw.text((x, y), ch, fill="black", font=font)
            y += font_size

    for idx, token in enumerate(tokens):
        x = 2
        y = (idx + 1) * cell_size + 5
        draw.text((x, y), token, fill="black", font=font)

    for i in range(num_tokens):
        for j in range(num_tokens):
            if matrix[i][j] == 1:
                x0 = (j + 1) * cell_size
                y0 = (i + 1) * cell_size
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill="black")

    return image

def get_coreference_labels(text, tokenizer,clusters_char_offsets, tokenizerx):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    encodingx = tokenizerx(text,  add_special_tokens=True)
    assert len(encodingx['input_ids']) == len(encodingx['input_ids'])
    offset_mapping = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 1. 收集所有 mention spans
    mention_spans = []  # (start_token_idx, end_token_idx) 左闭右开
    mention2cluster = {}  # mention_id -> cluster_id
    mention_id = 0

    # print("\n📌 Tokenized text:")
    # print(" ".join(tokens))
    # print()

    for cluster_id, cluster in enumerate(clusters_char_offsets):
        for mention in cluster:
            char_start, char_end = mention
            token_span = None
            for idx, (start, end) in enumerate(offset_mapping):
                if start >= end:
                    continue  # 跳过特殊 token 或空 token
                # 如果 token 和 mention 有字符重叠
                if not (end <= char_start or start >= char_end):
                    if token_span is None:
                        token_span = [idx, idx]
                    token_span[1] = idx
            if token_span:
                start_idx, end_idx = token_span[0], token_span[1] + 1  # 转换为左闭右开
                mention_spans.append((start_idx, end_idx))
                mention2cluster[mention_id] = cluster_id

                span_tokens = tokens[start_idx:end_idx]
                span_ids = input_ids[start_idx:end_idx]
                span_text = tokenizer.convert_tokens_to_string(span_tokens)

                # print(f"🔹 Mention {mention_id}:")
                # print(f"   Char Span:  {mention} → \"{text[char_start:char_end]}\"")
                # print(f"   Token Span: [{start_idx}:{end_idx}) → Tokens: {span_tokens}")
                # print(f"   Span Text:  \"{span_text}\"")
                # print()

                mention_id += 1


    # 构造 token2clusters 映射
    token2clusters = {}
    for mention_id, (start, end) in enumerate(mention_spans):
        cluster_id = mention2cluster[mention_id]
        for i in range(start, end):
            token2clusters.setdefault(i, set()).add(cluster_id)

    # 构造全 token coref 矩阵
    num_tokens = len(input_ids)
    token_coref_matrix = np.zeros((num_tokens, num_tokens), dtype=int)

    for i in range(num_tokens):
        for j in range(num_tokens):
            if i != j:
                if token2clusters.get(i) and token2clusters.get(j):
                    if token2clusters[i] & token2clusters[j]:
                        token_coref_matrix[i][j] = 1

    # print("\n✅ Token-level coreference matrix (size={}):".format(num_tokens))
    # print(token_coref_matrix)
    return token_coref_matrix , tokens

def  resolute_set(data_set, coref_model, tokenizer, tokenizerx):
    for data in tqdm.tqdm(data_set):

    # for i, data in tqdm.tqdm(enumerate(data_set)):
        # if i < 277:
        #     continue
        re = data['ref_exp']
        know = data['knowledge']
        all_tex =  re + '. ' + know
        coref_result = coref_model.predict(all_tex)
        clusters_char_offsets = coref_result['clusters_char_offsets']
        token_coref_matrix , tokens =  get_coreference_labels(all_tex,  tokenizer, clusters_char_offsets,tokenizerx)
        token_coref_matrix = torch.from_numpy(token_coref_matrix)
        data['coref_matrix'] = token_coref_matrix.tolist()
        # re_ids = tokenizer(re).input_ids
        # know_ids = tokenizer(know).input_ids
        #
        # refer_labels = torch.zeros(len(re_ids),len(know_ids))
        #
        #
        # if  len(re_ids) + len(know_ids) ==  token_coref_matrix.size(0)+1:
        #     refer_labels[:len(re_ids)-1, 1:] = token_coref_matrix[:len(re_ids)-1, len(re_ids): ]
        # else:
        #     cha_zhi = len(re_ids)+ len(know_ids) - token_coref_matrix.size(0)-1
        #     if cha_zhi >0:
        #         refer_labels[:len(re_ids) - 1, 1+cha_zhi:] = token_coref_matrix[:len(re_ids) - 1, len(re_ids):]
        #         refer_labels[:len(re_ids) -1, 1:1+cha_zhi] = token_coref_matrix[:len(re_ids)-1, len(re_ids):len(re_ids)+1].repeat(1,cha_zhi)
        #     else:
        #         refer_labels[:len(re_ids) - 1, 1:] = token_coref_matrix[:len(re_ids) - 1, len(re_ids)-cha_zhi:]
        #
        # data['coref_matrix'] = refer_labels.tolist()




        data['all_tex'] = all_tex
        # image = draw_coref_matrix_vertical_tokens(token_coref_matrix, tokens)
        # image.show()
    return  data_set

sk = json.load(open('/media/team/data/CODE/slef_code/final_code/mdetr/dataset/sk_vg/annotations.json'))
train_set = sk['train']
val_set = sk['val']
test_set = sk['test']

coref_model =Maverick( hf_name_or_path="/media/team/data/CODE/vg/GLIP-main/maverick-coref/weights",  device ="cuda:0",)

# tokenizer = RobertaTokenizerFast.from_pretrained(
#     '/media/team/data/CODE/transform/roabert', )
# tokenizer = BertTokenizerFast.from_pretrained(
#     '/media/team/data/CODE/transform/bert-base-uncased', )
tokenizer =  XLMRobertaTokenizerFast.from_pretrained('/media/team/data/CODE/transform/xlmroberta')
tokenizerx =  XLMRobertaTokenizer.from_pretrained('/media/team/data/CODE/slef_code/test_code/unifer_vg/pretrain_weights/beit3.spm')

test_set = resolute_set(test_set, coref_model,tokenizer,tokenizerx)
train_set = resolute_set(train_set, coref_model,tokenizer,tokenizerx)
val_set = resolute_set(val_set, coref_model,tokenizer,tokenizerx)


json.dump(test_set, open('xbert_anotations_cr_test.json', 'w'))
json.dump(train_set, open('xbert_anotations_cr_train.json', 'w'))
json.dump(val_set, open('xbert_anotations_cr_val.json', 'w'))

# json.dump(test_set, open('xbert_split_anotations_cr_test.json', 'w'))
# json.dump(train_set, open('xbert_split_anotations_cr_train.json', 'w'))
# json.dump(val_set, open('xbert_split_anotations_cr_val.json', 'w'))
