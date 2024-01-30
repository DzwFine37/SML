import argparse
import pickle

import numpy as np
from tqdm import tqdm

label = open(r'data/wlasl/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(r'reuslt/jointbest_acc.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(r'reuslt/bonebest_acc.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(r'reuslt/jmotionbest_acc.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open(r'reuslt/bmotionbest_acc.pkl', 'rb')
r4 = list(pickle.load(r4).items())
r5 = open(r'reuslt/jjmbest_acc.pkl', 'rb')
r5 = list(pickle.load(r5).items())
r6 = open(r'reuslt/bbmbest_acc.pkl', 'rb')
r6 = list(pickle.load(r6).items())

alpha = [0.7202576232110037, 0.8893489771693335, 0.48401579165552805, 0.3335250814870048, 0.9689584951972064, 0.8343688746910617]


right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0

with open(r'reuslt/predictions_rgb.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        name5, r55 = r5[i]
        name6, r66 = r6[i]
        assert name == name1 == name2 == name3 == name4 == name5 == name6
        # assert name == name5 == name6
        mean += r55.mean()

        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3] + r55*alpha[4] + r66*alpha[5]) / np.array(alpha).sum()
        # score = (r11*0.666677276294069 + r22*0.8009911656970481 + r33*0.505101293629485 + r44*0.3383621179081272 + r55*0.7895527356596431 + r66*0.6323503632754304) / np.array([0.666677276294069,0.8009911656970481,0.505101293629485,0.3383621179081272,0.7895527356596431,0.6323503632754304]).sum()
        # score = (r66*alpha[0] ) / np.array(alpha).sum()
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(score)
        scores.append(score)
        preds.append(r)
        right_num += int(r == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, r))
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)
#four_input 53.8 0.871
#six_input 55.67 0.886
#(0.666677276294069, 0.8009911656970481, 0.505101293629485, 0.3383621179081272, 0.7895527356596431, 0.6323503632754304)

f.close()
print(mean/len(label[0]))

# with open('./val_pred.pkl', 'wb') as f:
#     # score_dict = dict(zip(names, preds))
#     score_dict = (names, preds)
#     pickle.dump(score_dict, f)

# with open('./gcn_ensembled.pkl', 'wb') as f:
#     score_dict = dict(zip(names, scores))
#     pickle.dump(score_dict, f)