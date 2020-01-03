import os
import xlwt

book = xlwt.Workbook(encoding='utf-8')
sheet = book.add_sheet('sheet1')
head = ['model', 'dataset', 'emb', 'scorefuse', 'weight', 'epoch', 'angle', 'accuracy', 'std']    #表头
for h in range(len(head)):
    sheet.write(0, h, head[h])

root = "/data1/mandi.luo/work/FaceRotation/cjcode-2-v1/"
case = 'mainBig'
txt_root = os.path.join(root, case, 'model_output', 'FE_frontalization', 'MP')
txt_lists = [i for i in os.listdir(txt_root) if i.endswith('.txt')]

all_data = []
for txt_list in txt_lists:
    print(txt_list)
    if 'LFW' in txt_list:
        continue
    with open(os.path.join(txt_root, txt_list), 'r') as f:
        for line in f.readlines():
            data = {}
            data['model'] = "_".join(txt_list.split('_')[:-5])
            data['dataset'] = txt_list.split('_')[-5]
            data['emb'] = txt_list.split('_')[-4][3]
            data['scorefuse'] = txt_list.split('_')[-3][9]
            data['weight'] = txt_list.split('_')[-2][6]
            data['epoch'] = txt_list.split('_')[-1][5]
            data['angle'] = line.strip().split(' ')[-5]
            data['accuracy'] = line.strip().split(' ')[-3]
            data['std'] = line.strip().split(' ')[-1]
            all_data.append(data)


i = 1
for data in all_data:
    print(data)
    sheet.write(i, 0, data['model'])
    sheet.write(i, 1, data['dataset'])
    sheet.write(i, 2, data['emb'])
    sheet.write(i, 3, data['scorefuse'])
    sheet.write(i, 4, data['weight'])
    sheet.write(i, 5, data['epoch'])
    sheet.write(i, 6, data['angle'])
    sheet.write(i, 7, data['accuracy'])
    sheet.write(i, 8, data['std'])

    i+=1

excel_path = "/data1/mandi.luo/work/FaceRotation/statistics.xls"
book.save(excel_path)
