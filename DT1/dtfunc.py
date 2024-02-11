import math
import copy

class DT:
    def __init__(self):
        self.root = Node()

    def fit(self, X, y):
        tree = DT()
        tree.root.fit(X, y)
        return tree
    
    def print_tree(self):
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if(node.children != {}):
                print(node)
            for value in node.children:
                queue.append(node.children[value])

    def predict(self, point):
        current = self.root
        attributes = self.root.X.attributes
        while current.answer == "":
            index = attributes.index(current.select_attr)
            current = current.children[point[index]]
        return current.answer
    
    

class Node:
    def __init__(self):
        self.X = None
        self.y = None
        self.info = 0
        self.children = {}
        self.data = {}
        self.parent = ""
        self.select_attr = ""
        self.answer = ""
    
    def __str__(self):
        string = []
        if(self.parent != ""):
            string.append("From node - " + self.parent + "\n")
            string.append("**************************************\n")
        else:
            string.append("Start decision tree !\n")
            string.append("**************************************\n")
        if(self.children != {}): string.append(" " + self.select_attr + "?\n\n")

        
        for value in self.children:
            string.append(" " + value)
            if(self.children[value].children != {}):
                string.append(" go for " + self.children[value].select_attr)
            else: string.append(": " + self.children[value].answer)
            string.append("\n")
        string.append("**************************************\n")
        return ''.join(string)
        
    def fit(self, X, y, table=None):
        self.X = X
        self.y = y
        self.info = self.info_cal(self.y.column(0))
        if(table == None): table = self.data_init()
        self.data = copy.deepcopy(table)
        self.data = self.count(self.data)
        self.data = self.info_D(self.data)

        max_gain = 0
        for attr in self.data:
            if self.data[attr]["gain"] >= max_gain:
                self.select_attr = attr
                max_gain = self.data[attr]["gain"]
        
        for value in self.data[self.select_attr]["values"]:
            child = Node()
            child.parent = self.select_attr + ": " + value
            if(self.data[self.select_attr]["values"][value]["info"] == 0):
                values = self.data[self.select_attr]["values"][value]["classes"]
                child.answer = max(values, key=values.get)
            elif(len(self.X.attributes) == 1):
                values = self.data[self.select_attr]["values"][value]["classes"]
                child.answer = max(values, key=values.get)
            else:
                child_table = copy.deepcopy(table)
                child_table.pop(self.select_attr)
                new_X = X.select(X.equal(self.select_attr, value)).drop_col([self.select_attr])
                new_y = y.select(X.equal(self.select_attr, value))
                child.fit(new_X, new_y, child_table)
            self.children[value] = child

    def data_init(self):
        data = {}
        for i in range(len(self.X.attributes)):
            attr_name = self.X.attributes[i]
            data[attr_name] = {}
            data[attr_name]["values"] = {}
            values = list(set(self.X.column(i)))

            for j in range(len(values)):
                value_name = values[j]
                data[attr_name]["values"][value_name] = {}
                data[attr_name]["values"][value_name]["count"] = 0
                data[attr_name]["values"][value_name]["classes"] = {}
                for label in set(self.y.column(0)):
                    data[attr_name]["values"][value_name]["classes"][label] = 0
        return data

    def count(self, data):
        attributes = self.X.attributes
        rows = self.X.rows
        for n in range(len(rows)):
            attributes
            for i in range(len(attributes)):
                label = self.y.column(0)[n]
                data[attributes[i]]["values"][rows[n][i]]["classes"][label] += 1
                data[attributes[i]]["values"][rows[n][i]]["count"] += 1
        return data

    def info_D(self, data):
        attributes = self.X.attributes
        for i in range(len(attributes)):
            attr_name = attributes[i]
            attr_info = 0
            for value_name in data[attr_name]["values"]:
                info = 0
                space = data[attr_name]["values"][value_name]["count"]
                for label in data[attr_name]["values"][value_name]["classes"]:
                    event = data[attr_name]["values"][value_name]["classes"][label]
                    info += self.entropy(event, space)
                data[attr_name]["values"][value_name]["info"] = info = -1*info
                attr_info += space/len(self.X.rows)*info
            space = data[attr_name]["info"] = attr_info
            space = data[attr_name]["gain"] = self.info - attr_info
        return data
    
    def info_cal(self, y):
        classes = set(y)
        info = 0

        for c in classes:
            space = len(y)
            event = y.count(c)
            info += self.entropy(event, space)
        
        return -1*info
    
    def entropy(self, event, space):
        if (event == 0): return 0
        prob = event/space
        return prob*math.log(prob,2)
