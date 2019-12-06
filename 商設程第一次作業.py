while(1234):
	a = int(input())
	change = 100 - a
	ans = [] 
	coin = [50,10,5,1]
	for i in range(0,len(coin)):
		ans.append(change // coin[i])
		change -= ans[i] * coin[i]
	for j in ans:
		print(j)
