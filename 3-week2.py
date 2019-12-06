def compare(n,x,y,r,p):
    global m
    sumtreat=[[] for _ in range(n+1)]
    sum1=[]
    for i in range(n+1):
        sum=[]
        for j in range(n+1):
            L=0
            for h in range(m):
                d=(((i-x[h])**2)+((j-y[h])**2))**0.5
                L+=(p[h])*max(((r[h]-d)/r[h]),0)
            sum.append(L)
        sum1.append(max(sum))
        sumtreat[i].append(max(sum))
        sumtreat[i].append(i)
        sumtreat[i].append(sum.index(max(sum)))
    a=sum1.index(max(sum1))
    return sumtreat[a]
n,m=map(int,input().split())
x=list(map(int,input().split()))
y=list(map(int,input().split()))
r=list(map(int,input().split()))
p=list(map(int,input().split()))
answer=compare(n,x,y,r,p)
print(answer[1],answer[2])