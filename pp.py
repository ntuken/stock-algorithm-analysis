r = input()
rate = r[2:].split(",")
for i in range(0,len(rate)):
    rate[i] = int(rate[i])
a = max(rate[1],rate[2])
b = min(rate[1],rate[2])
rate[1] = a
rate[2] = b

hw = 0
mt1 = 0
mt2 = 0
f = 0
order = [hw,mt1,mt2,f]
data = []
for i in range(0,int(r[0])+1):
    n = input()
    n = n.split(",")
    if i == 0:
        for k in range(0,len(n)):
            if n[k] == "HW":
                hw += k
            elif n[k] == "MT1":
                mt1 += k
            elif n[k] == "MT2":
                mt2 += k
            elif n[k] == "F":
                f += k
    elif i > 0:
        for j in range(0,len(n)):
            n[j] = int(n[j])
        c = max(n[mt1],n[mt2]) 
        d = min(n[mt1],n[mt2]) 
        n[mt1] = c
        n[mt2] = d
    data.append(n)

g_rate = {}
g_rate[hw] = rate[0]
g_rate[mt1] = rate[1]
g_rate[mt2] = rate[2]
g_rate[f] = rate[3]

final = []
for i in range(1,len(data)):
    grade = 0
    for j in range(1,len(data[0])):
        grade += data[i][j]*g_rate[j]
    average = grade/100
    final.append(average)

b = max(final)
for i in range(0,len(final)):
    if final[i] == b:
        break
a = int(i)+1
b = int(b)
print(str(a)+","+str(b))