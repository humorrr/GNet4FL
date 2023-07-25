import csv
import glob
import pickle
import re
import time

import numpy as np
import javalang
import pandas as pd
from anytree import AnyNode
import scipy.sparse as sps
import os
import re

def getBugLine(project,num,start):
    path_bug = 'data/bug_line'
    items = []
    delinenum = []
    allbug_line = []
    linenum = []
    for i in range(start,num):
        bug_line_name = '{}-{}.buggy.lines'.format(project,i)
        cur_path_bug = os.path.join(path_bug, bug_line_name)
        one_bug = []
        with open(cur_path_bug, "r") as file:
            lines = file.readlines()
            line = lines[0]
            line = line.split("#")[0]
            java_name = line.split("/")[-1]
            # print("java_name:", java_name)
        with open(cur_path_bug, "r") as file:
            for line in file.readlines():
                # string = 'abe(ac)ad)'
                line1 = line.split("/")[-1]
                # line = line.split("#")[0]
                # name = line.split("/")[-1]
                # if name!=java_name:
                #     continue
                # print("line:",line1)
                p1 = re.compile(r'[#](.*?)[#]', re.S)
                bug_line = re.findall(p1, line)
                # print("bug:",bug_line)
                for item in bug_line:
                    temp = int(item)
                one_bug.append(temp)

        allbug_line.append(one_bug)
        # print(cur_path_code)
        string = ''
    return allbug_line

COMMENT_RX = re.compile("package.*|import.*", re.MULTILINE)
# COMMENT_RX2 = re.compile(r'\', re.MULTILINE)

def parse_java(code,lastid):
    import javalang
    from javalang.ast import Node

    def get_token(node):
        token = ''
        if isinstance(node, str):
            token = node.replace('\"|\'', '')
        elif isinstance(node, set):
            token = 'Modifier'  # node.pop()
        elif isinstance(node, Node):
            token = node.__class__.__name__

        return token

    def get_children(root):
        if isinstance(root, Node):
            children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def get_sequence(node, sequence):
        token, children = get_token(node), get_children(node)
        # print("token:",token)

        sequence.append(token)
        # sequence.append('222')
        for child in children:
            # print("child:", child)
            get_sequence(child, sequence)



    def createtree(root, node, nodelist,lastid, parent = None, position = None):
        id = len(nodelist)+lastid
        a=len(nodelist)
        # print(id)
        token, children = get_token(node), get_children(node)
        if hasattr(node, 'position'):
            if node.position != None:
                position = node.position
                # print("has!:", node.position, node)
        if  a== 0:
            root.token = token
            root.data = node
            root.id=id
            root.position=0
        else:
            newnode = AnyNode(id = id, token = token, data = node, parent = parent, position = position)
        nodelist.append(token)
        for child in children:
            if a == 0:
                createtree(root, child, nodelist,lastid, parent = root, position = position)
            else:
                createtree(root, child, nodelist,lastid, parent = newnode, position = position)

    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    seq = []
    get_sequence(tree, seq)
    result = ' '.join(seq)
    nodelist = []
    newtree = AnyNode(id = 0, token = None, data = None, position = None)
    createtree(newtree, tree, nodelist,lastid)
    return ' '.join(result.split()), nodelist, newtree


def hump2underline(hunp_str):
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p, r'\1 \2', hunp_str)
    return sub


def process_source(code):
    # code = code.replace('\n', ' ').strip()
    tokens = list(javalang.tokenizer.tokenize(code))
    tks = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            tks.append('_STR')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            tks.append('_NUM')
        elif tk.__class__.__name__ == 'Boolean':
            tks.append('_BOOL')
        else:
            tks.append(tk.value)
    return " ".join(tks)


def transformer(code,lastid):
    # code = COMMENT_RX.sub('', code)
    # code = COMMENT_RX2.sub("\"", code)
    # print("code:", code)
    # code = ' '.join([hump2underline(i) for i in process_source(code).split()])
    # print("len:",code)
    code_seq, sbt, tree = parse_java(code,lastid)
    return code_seq, sbt, tree

def getnodeandedge_astonly(node, nodeindexlist, vocabdict, src, tgt, po):
    token = node.token
    nodeindexlist.append([node.id, node.token])
    po.append(node.position)
    if  token == ' ClassDeclaration' or token=='MethodDeclaration' or token=='ConstructorDeclaration' or token =='ClassDeclaration'\
            or 'Statement' in token or 'TernaryExpression' in token or 'For' in token:
        # print("Declaration true!")
        for child in node.children:
            src.append(node.id)
            tgt.append(child.id)
            src.append(child.id)
            tgt.append(node.id)
            getnodeandedge_astonly(child, nodeindexlist, vocabdict, src, tgt, po)
    else:
        return


def get_file(path):
    return glob.glob(path + r'/*')

# PROJECTS = ['Lang', 'Chart', 'Math', 'Mockito', 'Time']
# # PROJECTS = ['Chart']
# PROJECT_BUGS = [
#     [str(x) for x in range(1, 66)],
#     [str(x) for x in range(1, 27)],
#     [str(x) for x in range(1, 107)],
#     [str(x) for x in range(1, 39)],
#     [str(x) for x in range(1, 28)]
# ]
PROJECTS = ['Chart']
PROJECT_BUGS = [
    [int(x) for x in range(1,27)]
]
line = []
node_num=0
id_data_label=[]
alldata=[]
allpo=[]
adjacent=[]
allx=[]
alledgesrc = []
alledgetgt=[]
alline=[]
treetoken_num=[]
allbuggy_lines=[]
alltimenum=[]
list_file = open('data/pkl/cov_{}.pickle'.format(PROJECTS[0]), 'rb')
times = pickle.load(list_file)
list_fileb = open('data/pkl/buggy_lines_{}.pickle'.format(PROJECTS[0]), 'rb')
bugliness = pickle.load(list_fileb)

time_start=time.time()

for project, bugs in zip(PROJECTS, PROJECT_BUGS):
    for bug in bugs:
        print(project,bug)
        path_code = 'data/code/{}/v{}'.format(project, bug)
        path_code = get_file(path_code)
        # print(path_code)
        with open(path_code[0], 'r', encoding = 'utf-8') as f:
            code = f.read()
        lastid = node_num
        code_seq, sbt, newtree = transformer(code,lastid)
        # sbt = list(set(sbt))
        vocabsize = len(sbt)
        tokenids = range(vocabsize)
        vocabdict = dict(zip(sbt, tokenids))
        # print("vocabdict:", vocabdict)
        x = []
        edgesrc = []
        edgetgt = []
        po = []
        getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt, po)
        adjacent.extend(edgesrc)
        allx.extend(x)
        alledgesrc.extend(edgesrc)
        alledgetgt.extend(edgetgt)
        node_num+=len(x)
        treetoken_num.append(len(x))
        for i in po:
            if i==0:
                line.append(0)
            else:
                line.append(i.line)

        alltoken = []
        for i in range(0, len(x)):
            if line[i] in times[bug-1]:
                alltimenum.append(times[bug-1][line[i]])
            else:
                alltimenum.append(-1)
            bugt = bug - 1
            if line[i] not in bugliness[bugt]:
                alltoken.append([x[i][0],x[i][1],0])
                allpo.append([0])
            else:
                alltoken.append([x[i][0],x[i][1], 1])
                allpo.append([1])
            alldata.append(x[i][1])
        alline.extend(line)
        id_data_label.extend(alltoken)
        line=[]
        # print(alltoken[1].data, alltoken[1].label)

# # feature
from gensim.models.word2vec import Word2Vec
sentence = []
for i in alldata:
    temp = []
    temp.append(i)
    sentence.append(temp)
model = Word2Vec(sentence,min_count = 1,vector_size=100)
# print(model.wv)
# model.save("node_w2v2.model")
# print(model)
vector = []
for item in alldata:
    vector.append(model.wv[item])
vector = np.array(vector)
print(vector)

nodeid=[]
# print(vector.shape)
for i in range(0,len(allx)):
    nodeid.append(allx[i][0])
print(len(alline))
list_file = open('data/pkl/linepo_{}.pickle'.format(project), 'wb')
pickle.dump(alline, list_file)
list_file.close()

treetoken_num
list_file2 = open('data/pkl/treetoken_num_{}.pickle'.format(project), 'wb')
pickle.dump(treetoken_num, list_file2)
list_file2.close()


nodeidx = np.array(nodeid)
c = np.insert(vector, 0, values = nodeid, axis = 1)
with open('data/csv/feature_{}.csv'.format(project), 'w', newline = '')as f:
    f_csv = csv.writer(f)
    for i in c:
        f_csv.writerow(i)
# ——————————
# # y
with open('data/csv/y_{}.csv'.format(project), 'w',newline='')as f:
    f_csv = csv.writer(f)
    for i in allpo:
        f_csv.writerow(i)

# ——————————————————
# matrix
length = int(len(alledgesrc) / 2)
edge = []
for i in range(0, len(alledgesrc)-1,2):
    tedg = [alledgesrc[i], alledgesrc[i + 1]]
    edge.append(tedg)
    i += 2
edge = np.array(edge)
with open('data/csv/edgecites_{}.csv'.format(project), 'w',newline='')as f:
    f_csv = csv.writer(f)
    # f_csv.writerow(headers)
    f_csv.writerows(edge)
