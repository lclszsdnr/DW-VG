import json
val = json.load(open('xbert_anotations_cr_val.json'))
test = json.load(open('xbert_anotations_cr_test.json'))
train = json.load(open('xbert_anotations_cr_train.json'))


new = {"train": train, "val": val, "test":test}

json.dump(new, open('xbert_anotations_cr_new.json', 'w'))
