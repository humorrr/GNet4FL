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
# COMMENT_RX = re.compile("package.*|import.*|(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/", re.MULTILINE)
from sklearn.manifold import TSNE

# from paper.draw_tsne import plot_with_labels

COMMENT_RX = re.compile("package.*|import.*", re.MULTILINE)
# COMMENT_RX2 = re.compile(r'\', re.MULTILINE)

class Etokeninfo:
    def __init__(self,id, data, label):
        self.id=id
        self.data = data
        self.label = label

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


# def process_source(code):
#     # code = code.replace('\n', ' ').strip()
#     tokens = list(javalang.tokenizer.tokenize(code))
#     tks = []
#     for tk in tokens:
#         if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
#             tks.append('_STR')
#         elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
#             tks.append('_NUM')
#         elif tk.__class__.__name__ == 'Boolean':
#             tks.append('_BOOL')
#         else:
#             tks.append(tk.value)
#     return " ".join(tks)


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
