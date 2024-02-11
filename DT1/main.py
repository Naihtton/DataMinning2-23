import datasfunc 
import dtfunc

data = datasfunc.read_csv("datasets/iris_processed.csv")

X, y = data.exclass("class")

tree = dtfunc.DT()
tree = tree.fit(X, y)

#tree.print_tree()

print(tree.predict(["<=5.4","2.8 -> 3.6","<=3","<=0.9"]))
print(tree.predict([">6.5","2.8 -> 3.6","3 -> 4.9","0.9 -> 1.7"]))
print(tree.predict([">6.5","2.8 -> 3.6",">4.9",">1.7"]))


