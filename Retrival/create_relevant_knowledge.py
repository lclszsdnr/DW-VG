import json
import copy

dataset = json.load(open('/media/team/data/CODE/slef_code/final_code/mdetr/dataset/sk_vg/annotations_new_refered.json'))
tranin_datas = dataset['train']
val_datas = dataset['val']
test_datas = dataset['test']

new_refers_train = json.load(open('/media/team/data/CODE/slef_code/final_code/mdetr/new_refers_train.json'))
new_refers_test = json.load(open('/media/team/data/CODE/slef_code/final_code/mdetr/new_refers_test.json'))

knowledge_map_train = json.load(open('/media/team/data/CODE/slef_code/final_code/mdetr/relevant_knowledge_map_train.json'))
knowledge_map_test = json.load(open('/media/team/data/CODE/slef_code/final_code/mdetr/relevant_knowledge_map_test.json'))



print(len(tranin_datas))
print(len(new_refers_train))
print(len(test_datas))
print(len(new_refers_test))



for i , new in enumerate(new_refers_train):
    tranin_datas[i]['ref_exp'] = new


for j , new in enumerate(new_refers_test):
    test_datas[j]['ref_exp'] = new

json.dump(dataset, open("annotations_relevant.json", 'w'))












# for i , new in enumerate(new_refers_train):
#     tranin_datas[i]['relevant_knowledge'] =( new.split(tranin_datas[i]['ref_exp']+' '+":")[-1]).split(tranin_datas[i]['ref_exp']+' '+",")[-1]
#     tranin_datas[i]['knowledge_map'] = knowledge_map_train[i]
#     if len( new.split(tranin_datas[i]['ref_exp']+' '+":")) ==1:
#         tranin_datas[i]['is_include'] =0
#     else:
#         tranin_datas[i]['is_include'] =1
#
#
# for j , new in enumerate(new_refers_test):
#     test_datas[j]['relevant_knowledge'] =( new.split(test_datas[j]['ref_exp']+' '+":")[-1]).split(test_datas[j]['ref_exp']+' '+",")[-1]
#     test_datas[j]['knowledge_map'] = knowledge_map_test[j]
#     if len( new.split(test_datas[j]['ref_exp']+' '+":")) ==1:
#         test_datas[j]['is_include'] =0
#     else:
#         test_datas[j]['is_include'] =1
# json.dump(dataset, open("o_annotations_relevant.json".format(i), 'w'))
# for i , new in enumerate(new_refers_train):
#     # tranin_datas[i]['relevant_knowledge'] =( new.split(tranin_datas[i]['ref_exp']+' '+":")[-1]).split(tranin_datas[i]['ref_exp']+' '+",")[-1]
#     # tranin_datas[i]['knowledge_map'] = knowledge_map_train[i]
#     if len( new.split(' '+":")) ==2:
#         new_data = copy.deepcopy(tranin_datas[i])
#         new_data['ref_exp'] = ( new.split(' '+":")[-1])
#         tranin_datas.append(new_data)
#     else:
#         new_data = copy.deepcopy(tranin_datas[i])
#         new_data['ref_exp'] = new
#         tranin_datas.append(new_data)


# json.dump(dataset, open("new_annotations.json".format(i), 'w'))
