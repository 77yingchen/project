import numpy as np
import csv

train1 = np.load('x_train.npy')
train2 = np.load('x_test.npy')

data = np.concatenate((train1, train2), axis=0)

all_item = []

with open('itemSet5.txt') as file:
	for l in file:
		if (l != '\n'):
			all_item.append(l.strip())

output = []

print data.shape
for i in range(0, 132):
	for j in range(0, 132):
		print (i, j)
		both_count = 0.0
		P2_unique_count = 0.0
		P1_unique_count = 0.0
		# we cannot make two same items as a pair
		if (i != j):
			for m in range(0, data.shape[0]):
				if (data[m][i] == 1 and data[m][j] == 1):
					both_count = both_count+1
				if (data[m][i] != 1 and data[m][j] == 1):
					P2_unique_count = P2_unique_count+1
				if (data[m][i] == 1 and data[m][j] != 1):
					P1_unique_count = P1_unique_count+1


			support = float(both_count)/data.shape[0]
			P2 = (float(both_count)+float(P2_unique_count))/data.shape[0]
			P1 = (float(both_count)+float(P1_unique_count))/data.shape[0]
			confidence = float(support)/float(P2)
			lift = float(confidence)/float(P1)
			# print both_count
			# print lift

			tmp = [[str(all_item[i]) + ' & ' + str(all_item[j])]]
			tmp.append(lift)
			tmp.append(confidence)
			output.append(tmp)
			

with open("lift_scores.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(output)





