from math import log
import re
import os

hamPath = "ham/"
spamPath = "spam/"
testPath = "test/"
hamDict = dict()
spamDict = dict()

# Hyperparameters for alpha and vocabulary size
alpha = 0.1
vocab = 20000

# Return email and word count in the given class
def countWordAndEmail(givenPath, givenDict):
    totalEmailCount = 0
    totalWordCount = 0
    for fileName in os.listdir(givenPath):
        totalEmailCount += 1
        with open(os.path.join(givenPath, fileName),'r') as f:
            for line in f:
                word = line.strip('\n')
                totalWordCount += 1
                if word in givenDict:
                    givenDict[word] += 1
                else:
                    givenDict[word] = 1
        f.close()
    return [totalEmailCount, totalWordCount]

hams = countWordAndEmail(hamPath, hamDict)
totalHamEmailCount = hams[0]
totalHamWordCount = hams[1]
spams = countWordAndEmail(spamPath, spamDict)
totalSpamEmailCount = spams[0]
totalSpamWordCount = spams[1]

# Smooth P(unseen) for ham and spam with log
logUnseenHam = log(alpha / (totalHamWordCount + (alpha * vocab)))
logUnseenSpam = log(alpha / (totalSpamWordCount + (alpha * vocab)))

# Compute log prior
totalEmailCount = totalHamEmailCount + totalSpamEmailCount
logPriorHam = log(totalHamEmailCount / totalEmailCount)
logPriorSpam = log(totalSpamEmailCount / totalEmailCount)

# Compute log posterior P(Wi|C)
for word, count in hamDict.items():
    hamDict[word] = log((hamDict[word] + alpha) / (totalHamWordCount + alpha * vocab))
for word, count in spamDict.items():
    spamDict[word] = log((spamDict[word] + alpha) / (totalSpamWordCount + alpha * vocab))

# Classify test emails into ham or spam
classified = dict()
for fileName in os.listdir(testPath):
    fileNameNum = re.search(r'\d+', fileName).group(0)
    logHamProb = logPriorHam
    logSpamProb = logPriorSpam
    with open(os.path.join(testPath, fileName), 'r') as f:
        for line in f:
            word = line.strip('\n')
            if word in hamDict:
                logHamProb += hamDict[word]
            else:
                logHamProb += logUnseenHam
            if word in spamDict:
                logSpamProb += spamDict[word]
            else:
                logSpamProb += logUnseenSpam
        if (logHamProb >= logSpamProb):
            classified[fileNameNum] = 0 # represent ham
        else:
            classified[fileNameNum] = 1 # represent spam

# Evaluate performance with metrics
TP = 0
FP = 0
FN = 0
TN = 0
truthSpam = dict()
with open("truthfile", 'r') as f:
    for line in f:
        truthSpamNum = line.strip('\n')
        truthSpam[truthSpamNum] = 1
for email in classified:
    if email in truthSpam:
        if classified[email] == 1:
            TP += 1
        else:
            FN += 1
    else:
        if classified[email] == 0:
            TN += 1
        else:
            FP += 1

accuracy_score = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = (2 * precision * recall) / (precision + recall)

print('[ Hyperparameters ]')
print('Alpha: \t{}'.format(alpha))
print('Vocab:\t{}'.format(vocab))
print('-'*20)
print('[ Confusion Matrix ]')
print('TP: {} \t FP: {} \nFN: {} \t TN: {}'.format(TP, FP, FN, TN))
print('-'*20)
print('[ Evaluation Metrics ]')
print('Accuracy: \t {:10.6f}'.format(accuracy_score))
print('Precision: \t {:10.6f}'.format(precision))
print('Recall: \t {:10.6f}'.format(recall))
print('F1-score: \t {:10.6f}'.format(f1_score))

# Write to report with evaluation metrics to 6 digits after the decimal
with open("report.txt", 'w', encoding='utf-8') as fp:
    fp.write('[ Hyperparameters ]\n')
    fp.write('Alpha: \t{}\n'.format(alpha))
    fp.write('Vocab:\t{}\n'.format(vocab))
    fp.write('-'*20 + '\n')
    fp.write('[ Confusion Matrix ]\n')
    fp.write('TP: {} \t FP: {} \nFN: {} \t TN: {}\n'.format(TP, FP, FN, TN))
    fp.write('-'*20 + '\n')
    fp.write('[ Evaluation Metrics ]\n')
    fp.write('Accuracy:\t{:10.6f}\n'.format(accuracy_score))
    fp.write('Precision:\t{:10.6f}\n'.format(precision))
    fp.write('Recall: \t{:10.6f}\n'.format(recall))
    fp.write('F1-score:\t{:10.6f}\n'.format(f1_score))