#โครงสร้าง 3 input nodes, 2 hidden nodes, 1 output node
#มี input ชุดเดียว คำนวณ backprop 1 ครั้งเพื่อปรับค่า
#import numpy 
from NNfunction import * # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction

X=([1,0,1,1])
W4=([0.2,0.4,-0.5,-0.4]) #weight ที่เกี่ยวข้องกับ node 4 [w14,w24,w34,bias4]
W5=([-0.3,0.1,0.2,0.2]) #weight ที่เกี่ยวข้องกับ node 5 [w15,w25,w35,bias5]
d6=1 #desire output
l=-0.9 # learning rate
#forward pass
print("\n-----Forward pass-----> ")
o4=Nout(X,W4) #call NNfunction
y4=sigmoid(o4) #call NNfunction
print("\nSum(V) of node 4 is: %8.3f, Y from node 4 is: %8.3f" % (o4,y4))

o5=Nout(X,W5)
y5=sigmoid(o5) #คำนวน y5
print("\nSum(V) of node 5 is: %8.3f, Y from node 5 is: %8.3f" % (o5,y5)) #print ผลลัพธ์


X6=([y4,y5,1]) # กำหนด input
W6=([-0.3,-0.2,0.1]) #weight ที่เกี่ยวข้องกับ node 6 [w46,w56,bias6]
o6=Nout(X6,W6)
y6=sigmoid(o6)
print("\nSum(V) of node 6 is: %8.3f, Y from node 6 is: %8.3f" % (o6,y6)) #print ผลลัพธ์


#backpropagation
#node 6
print("\n<---- Back propagation & calculate new Weights and Biases ----")
e6=d6-y6
g6=gradOut(e6,y6) #call NNfunction
dw46=deltaw(l,g6,y4) #call NNfunction
w46n=-0.3+dw46
dw56=deltaw(l,g6,y5)
w56n=-0.2+dw56
db6=deltaw(l,g6,1)
b6n=0.1+db6
print("\nNew w46 is %8.3f, New w56 is:%8.3f, New bias 6 is %8.3f"% (w46n,w56n,b6n))

#node5
#pre gradient5=g6*w56
sumN6w=g6*(-0.2)
g5=gradH(y5,sumN6w)
dw15=deltaw(l,g5,1)
w15n=-0.3+dw15
dw25=deltaw(l,g5,0)
w25n=0.1+dw25
dw35=deltaw(l,g5,1)
w35n=0.2+dw35
db5=deltaw(l,g5,1)
b5n=0.2+db5
print("\nNew w15 is %8.3f, New w25 is:%8.3f,New w35 is:%8.3f, New bias 6 is %8.3f"% (w15n,w25n,w35n,b5n))

#node4
#pre gradient4=g6*w46
sumN6w=g6*(-0.3)
g4=gradH(y4,sumN6w)
dw14=deltaw(l,g5,1)
w14n=0.2+dw14
dw24=deltaw(l,g5,0)
w24n=0.4+dw24
dw34=deltaw(l,g5,1)
w34n=-0.5+dw34
db4=deltaw(l,g5,1)
b4n=-0.4+db4
print("\nNew w15 is %8.3f, New w25 is:%8.3f,New w35 is:%8.3f, New bias 6 is %8.3f"% (w14n,w24n,w34n,b4n))
