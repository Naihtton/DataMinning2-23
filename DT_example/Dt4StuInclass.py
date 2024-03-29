from scipy import linalg
import numpy as np
from numpy import loadtxt
import math
from Dtreefunc import *
f=open("buycom.txt","r")
X=f.readlines()

#เตรียม พื้นที่เพื่อเก็บผลการคำนวณแยกตาม class
L=2 #col (2 class)
N=3 #col(3 class)
M=3 #row

age=np.zeros(3)
ageCI=[[0 for i in range(M)] for j in range(N)] # zero matrix 3 rows 3 columns (class and info gain of age)
# ให้นศ ทำ zero matrix 3 rows 3 columns (class and info gain of income)
income =  np.zeros(3)
incomeCI = [[0 for i in range(M)] for j in range(N)]

credit=np.zeros(2)
creditCI=[[0 for i in range(M)] for j in range(L)]

stu=np.zeros(2)
stuCI=[[0 for i in range(M)] for j in range(L)]# zero matrix 3 rows 2 columns (class and info gain of student)
# ให้นศ ทำ zero matrix 3 rows 3 columns (class and info gain of credit)

buy = np.zeros(2)
buyCI = [[0 for i in range(M)] for j in range(L)]

#วน loop เพื่อนับข้อมูล แยกตามรายละเอียด attb และ class
for i in range(0,15):
    if ((X[i].count('<=30')==1)): 
        age[0]+=1 # total sample tha age <=30
        if ((X[i].count('<=30')==1)) and (X[i].count('No')==1):
            ageCI[0][0]+=1 #class no
        else:
            ageCI[0][1]+=1 #class yes
    elif(X[i].count('31-40')==1):
        age[1]+=1
        if ((X[i].count('31-40')==1)) and (X[i].count('No')==1):
            ageCI[1][0]+=1
        else:
            ageCI[1][1]+=1
    elif(X[i].count('>=40')==1):
        age[2]+=1
        if ((X[i].count('>=40')==1)) and (X[i].count('No')==1):
            ageCI[2][0]+=1
        else:
            ageCI[2][1]+=1

    if (X[i].count('low')==1):
        income[0]+=1
        if ((X[i].count('low')==1)) and (X[i].count('No')==1):
            incomeCI[0][0]+=1 #class no
        else:
            incomeCI[0][1]+=1 #class yes
    elif(X[i].count('medium')==1):
        income[1]+=1
        if ((X[i].count('medium')==1)) and (X[i].count('No')==1):
            incomeCI[1][0]+=1
        else:
            incomeCI[1][1]+=1
    elif(X[i].count('high')==1):
        income[2]+=1
        if ((X[i].count('high')==1)) and (X[i].count('No')==1):
            incomeCI[2][0]+=1
        else:
            incomeCI[2][1]+=1
            
    if ((X[i].count('s_no')==1)): 
        stu[0]+=1 # total sample tha age <=30
        if ((X[i].count('s_no')==1)) and (X[i].count('No')==1):
            stuCI[0][0]+=1 #class no
        else:
            stuCI[0][1]+=1 #class yes
    elif(X[i].count('s_yes')==1):
        stu[1]+=1
        if ((X[i].count('s_yes')==1)) and (X[i].count('No')==1):
            stuCI[1][0]+=1
        else:
            stuCI[1][1]+=1
    
        #ให้นักศึกษาทำต่อจนครบทุก Attb ในห้องเรียน
    

    if (X[i].count('No')==1):
        buy[0]+=1
    elif(X[i].count('Yes')==1):
        buy[1]+=1


# calculate information gain of dataset and attb
# info D,age,income,stu,credit
info = np.zeros(4)
InD=entropy(buy[1],buy[0])

ageCI[0][2]=entropy(ageCI[0][0],ageCI[0][1]) 
ageCI[1][2]=entropy(ageCI[1][0],ageCI[1][1])
ageCI[2][2]=entropy(ageCI[2][0],ageCI[2][1])

incomeCI[0][2]=entropy(incomeCI[0][0],incomeCI[0][1])
incomeCI[1][2]=entropy(incomeCI[1][0],incomeCI[1][1])
incomeCI[2][2]=entropy(incomeCI[2][0],incomeCI[2][1])

stuCI[0][2]= entropy(stuCI[0][0],stuCI[0][1])
stuCI[1][2]= entropy(stuCI[1][0],stuCI[1][1])

creditCI[0][2]= entropy(creditCI[0][0],creditCI[0][1])
creditCI[1][2]= entropy(creditCI[1][0],creditCI[1][1])

# หาค่า gain แบบไม่ใช้ และใช้ฟังก์ชัน
"""
การหาแบบไม่ใช้ฟังก์ชัน
Info_ageD = ((age[0]/14)*ageCI[0][2])+((age[1]/14)*ageCI[1][2])+((age[2]/14)*ageCI[2][2])
print("InfoD age is",Info_ageD)
print("Age Ci [:],[2] is",[ageCI[0][2],ageCI[1][2],ageCI[2][2]])
print("InfoD age is",Info_ageD)
"""
Info_ageD = inforD(age,[ageCI[0][2],ageCI[1][2],ageCI[2][2]])
Info_incomeD = inforD(income,[incomeCI[0][2],incomeCI[1][2],incomeCI[2][2]])
Info_studentD = inforD(stu,[stuCI[0][2],stuCI[1][2]])
Info_creditD = inforD(credit,[creditCI[0][2],creditCI[1][2]])

# แสดงผลการทำงานรอบแรก
"""
print("Age count is", age)
print("income count is",income)
print("student count is",stu)
print("credit rating count is",credit)
print("Buy computer count is",buy)
print("Age Info relate to class",ageCI)z
print("Income Info relate to class",incomeCI)
print("Student Info relate to class",stuCI)
print("Credit rating Info relate to class",creditCI)

print("Info(D) is %5.3f" % InD)
print("Info(age <=30(2,3) is %5.3f" % ageCI[0][2])
print("Info(age 31...40(4,0) is %5.3f" % ageCI[1][2])
print("Info(age >40 (3,2) is %5.3f" % ageCI[2][2])

print("Info(income low(1,3) is %5.3f" % incomeCI[0][2])
print("Info(income medium(2,4) is %5.3f" % incomeCI[1][2])
print("Info(income high(2,2) is %5.3f" % incomeCI[2][2])

print("Info(student No (4,3) is %5.3f" % stuCI[0][2])
print("Info(student Yes (1,6) is %5.3f" % stuCI[1][2])

print("Info(credit fair(2,6) is %5.3f" % creditCI[0][2])
print("Info(credit excellent(3,3) is %5.3f" % creditCI[1][2])
print("Info age (D) is %5.3f" % Info_ageD)
print("Info income (D) is %5.3f" % Info_incomeD)
print("Info student (D) is %5.3f" % Info_studentD)
print("Info credit rating (D) is %5.3f" % Info_creditD)
"""
print("\n***Gain results of all dataset***")
gainAge=InD-Info_ageD
print("Gain (age) is %5.3f"% gainAge)
gainIn=InD-Info_incomeD
print("Gain (Income) is %5.3f"% gainIn)
gainStu=InD-Info_studentD
print("Gain (Student) is %5.3f"% gainStu)
gainCre=InD-Info_creditD
print("Gain (Credit rating) is %5.3f"% gainCre)

#rule of root node

Result_All=[gainAge,gainIn,gainStu,gainCre]
max_gain=max(Result_All)
pos=np.argmax(Result_All)
print("max gain of attb is %5.3f" % max_gain,"position is",pos)

#วน loop แยก dataset ตาม attb age
X2L=[] #ข้อมูลสำหรับสร้าง level 2 ที่ Age <=30
X2M=[] #ข้อมูลสำหรับสร้าง level 2 ที่ Age 31-40
X2R=[] #ข้อมูลสำหรับสร้าง level 2 ที่ Age >=40
f1=open("buycomL2left.txt","w")
f2=open("buycomL2middle.txt","w")
f3=open("buycomL2right.txt","w")

for i in range(0,15):
    if ((X[i].count('<=30')==1)): 
        f1.write(str(X[i]))
    
    elif(X[i].count('31-40')==1):
        f2.write(str(X[i]))
        
    elif(X[i].count('>=40')==1):
        f3.write(str(X[i]))

# dataset of layer 2 of dtree generate
f1=open("buycomL2left.txt","r")
f2=open("buycomL2middle.txt","r")
f3=open("buycomL2right.txt","r")
X2L=f1.readlines()
X2M=f2.readlines()
X2R=f3.readlines()


# recursive line 14 สำหรับการสร้าง tree ชั้นที่ 2 สำหรับ dataset age<=30


# วน loop เพื่อนับข้อมูล แยกตามรายละเอียด attb และ class
# data ที่ age <=30



# calculate information gain of dataset and attb
# info D,age,income,stu,credit


# หาค่า gain แบบใช้ฟังก์ชัน



# แสดงผลการทำงาน รอบ2 ฝั่งซ้าย




# recursive line 14 สำหรับการสร้าง tree ชั้นที่ 2 สำหรับ dataset age>40


# วน loop เพื่อนับข้อมูล แยกตามรายละเอียด attb และ class
# data ที่ age >40



# calculate information gain of dataset and attb
# info D,age,income,stu,credit


# หาค่า gain แบบใช้ฟังก์ชัน



# แสดงผลการทำงานรอบสอง ฝั่งขวา




#สร้าง tree
#rule extraction
#model evaluation