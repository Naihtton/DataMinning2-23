import csv


class datas:
    def __init__(self, attributes, rows):
        self.attributes = attributes
        self.rows = rows
        self.num = len(rows) - 1

    def column(self, number):
        return [row[number] for row in self.rows]

    def equal(self, attr_name, value_name):
        index = self.attributes.index(attr_name)
        values = self.column(index)
        return [i for i in range(len(values)) if values[i] == value_name]
    
    def excl(self, list, indices):
        return [list[i] for i in range(len(list)) if i not in indices]

    def incl(self, list, indices):
        return [list[i] for i in range(len(list)) if i in indices]

    def drop_row(self, indices):
        return datas(self.attributes, self.excl(self.rows, indices))

    def drop_col(self, list_of_column):
        tabel = datas(self.attributes, self.rows)
        if isinstance(list_of_column, list) and all(isinstance(column, str) for column in list_of_column):
            list_of_column = [tabel.attributes.index(column) for column in list_of_column]
        tabel.attributes = [self.attributes[i] for i in range(len(self.attributes)) if i not in list_of_column]
        tabel.rows = [[item for item in tabel.excl(row, list_of_column)] for row in tabel.rows]
        return tabel

    def select(self, indices):
        return datas(self.attributes, self.incl(self.rows, indices))

    def exclass(self, attr_name):
        index = self.attributes.index(attr_name)
        y = datas(self.attributes[index], [[row[index]] for row in self.rows])
        X = datas(self.excl(self.attributes, [index]), self.drop_col([index]).rows)
        return X, y





def read_csv(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    attributes = rows[0]
    data = rows[1:]

    return datas(attributes, data)

