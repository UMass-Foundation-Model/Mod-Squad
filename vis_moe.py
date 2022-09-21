import torch
import numpy as np; np.random.seed(0)
import seaborn as sns; #sns.set_theme()
import matplotlib.pyplot as plt
import collections

the_list = torch.load('vis.t7')
print(the_list)

# Layer, img_type, Expert

def vis_img_to_expert(data, depth):
    data = collections.OrderedDict(sorted(data.items()))
    task_name = []
    the_data = []
    for _key, expert in data.items():
        task_name.append(_key)
        the_data.append(expert)
    the_data = np.array(the_data)

    ax = sns.heatmap(the_data, cmap='Blues', yticklabels=task_name)
    ax.set_title('Layer '+str(depth))
    plt.tight_layout()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig('vis/Layer '+str(depth) +'.pdf')
    print('save to ', 'vis/Layer '+str(depth) +'.pdf')
    plt.close()

    # task_num = len(task_name)
    # distance = np.zeros((task_num, task_num))
    # relation = []
    # for i in range(0, task_num):
    #     for j in range(i+1, task_num):
    #         distance[i,j] = (the_data[i, :]/100 * the_data[j, :]/100).sum() 
    #         distance[j, i] = distance[i, j]
    #         relation.append([distance[i,j], task_name[i], task_name[j]])

    # relation = sorted(relation, key=lambda item: item[0], reverse=True)
    # for i in range(10):
    #     print('Top ',i, ' ', relation[i][1], ' ', relation[i][2], ' ', relation[i][0])
    # # print(relation)
    # ax = sns.heatmap(distance, cmap='Blues', yticklabels=task_name, xticklabels=task_name)
    # ax.set_title('Layer '+str(depth))
    
    # # plt.show()

def vis_all_task_relation(the_list):
    all_dict = the_list[0]
    task_num = len(all_dict.keys())
    distance = np.zeros((task_num, task_num))
    for depth in range(0, 12):
        all_dict = collections.OrderedDict(sorted(the_list[depth].items()))
        task_name = []
        the_data = []
        for _key, expert in all_dict.items():
            task_name.append(_key)
            the_data.append(expert)
        print(task_name)

        the_data = np.array(the_data)

        task_num = len(task_name)
        for i in range(0, task_num):
            for j in range(i+1, task_num):
                distance[i,j] = distance[i,j] + (the_data[i, :]/100 * the_data[j, :]/100).sum() 
                distance[j, i] = distance[i, j]

    relation = []
    for i in range(0, task_num):
        for j in range(i+1, task_num):
            relation.append([distance[i,j], task_name[i], task_name[j]])

    relation = sorted(relation, key=lambda item: item[0], reverse=True)
    for i in range(10):
        print('Top ',i, ' ', relation[i][1], ' ', relation[i][2], ' ', relation[i][0])
    # print(relation)
    ax = sns.heatmap(distance, annot=True, cmap='Blues', yticklabels=task_name, xticklabels=task_name)
    ax.set_title('all')
    plt.tight_layout()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    # plt.show()
    plt.savefig('vis/task_relation_all.pdf')
    plt.close()


for depth in range(0,12):
    vis_img_to_expert(the_list[depth], depth)
vis_all_task_relation(the_list)
