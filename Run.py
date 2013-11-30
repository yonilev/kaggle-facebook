import unicodecsv
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import data,FreqDist
from bs4 import BeautifulSoup
from random import randint
from sklearn.feature_extraction.text import FeatureHasher
from sklearn.linear_model import Perceptron,SGDClassifier
from itertools import izip
import cPickle
from time import time
import os
from sklearn.metrics import classification_report,precision_recall_fscore_support
import operator
from scipy.sparse import hstack
from pygments.lexers import guess_lexer




def createStopWords():
    return set(stopwords.words('english'))

class SampleAttributes:
    def __init__(self):
        self.hasCode = False
        self.hasImage = False
        self.hasList = False
        self.hasLink = False
        self.hasUrl = False
        self.hasFormula = False
    def __repr__(self):
        return 'hasCode:{0} , hasImage:{1} , hasList:{2} , hasLink:{3} , hasUrl:{4} , hasFormula:{4}'.format(self.hasCode,self.hasImage,self.hasList,self.hasLink,self.hasFormula)

def parseSample(sample):
    codeTokens = set()
    formulaTokens = set()
    codeNames = set()
    att = SampleAttributes()
    formulaRegex = '''\$.+?\$\$?'''
    urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    title = sample[1]
    
    prev = title
    title = re.sub(urlRegex,'',title)
    if len(prev)!=len(title):
        att.hasUrl = True 
    
    prev = title
    title = re.sub(formulaRegex,'',title)
    if len(prev)!=len(title):
        att.hasFormula = True 
           
    title = re.sub('\n+','\n',title)
    title = re.sub(' +',' ',title)

    body = sample[2]
    soup = BeautifulSoup(body)
    codeNodes = soup.findAll('code')
    for codeNode in codeNodes:
        code = u''.join(codeNode.findAll(text=True))
        att.hasCode = True
        lexer = guess_lexer(code)
        lexerName = re.sub(' +','',lexer.name)
        codeNames.add((lexerName))
        temp = code.lower().split()
        for w in temp:
            if LEXICON_CODE.contains(w):
                codeTokens.add(w)
        codeNode.extract()
    if soup.a:
        att.hasLink = True
    if soup.ul or soup.ol:
        att.hasList = True
    if soup.img:
        att.hasImage = True
    cleanBody = u''.join(soup.findAll(text=True))
    prev = cleanBody
    cleanBody = re.sub(urlRegex,'',cleanBody)
    if len(prev)!=len(cleanBody):
        att.hasUrl = True
        
    prev = cleanBody
    matches = re.findall(formulaRegex,cleanBody)
    if matches:
        for m in matches:
            m.replace('$','')
            temp = m.split()
            for w in temp:
                if LEXICON_FORMULA.contains(w):
                    formulaTokens.add(w)
                
    cleanBody = re.sub(formulaRegex,'',cleanBody)
    if len(prev)!=len(cleanBody):
        att.hasFormula = True 

    return title,cleanBody,att,formulaTokens,codeTokens,codeNames

    
def parseHtmlSamples(fileName):
    if not os.path.exists('data/parsed'):
        os.makedirs('data/parsed')
    reader = unicodecsv.reader(open('data/raw/'+fileName+'.csv', 'r'))
    idsFile = open('data/parsed/'+fileName+'Ids','w')
    sampleFile = unicodecsv.writer(open('data/parsed/'+fileName+'Samples.csv', 'w'))
    tagsFile = open('data/parsed/'+fileName+'Tags','w')
    i = -1
    for row in reader:
        i+=1
        if i==0:
            continue
        
        if i>1:
            idsFile.write('\n')
            tagsFile.write('\n')
        title,cleanBody,att,formulaTokens,codeTokens,codeNames = parseSample(row)    
        if len(row)==4:
            tags = row[3]
        else:
            tags = ''
        idsFile.write(row[0])
        sampleFile.writerow([title,cleanBody,att.hasCode,att.hasImage,att.hasList,att.hasLink,att.hasFormula,att.hasUrl,' '.join(formulaTokens),' '.join(codeTokens),' '.join(codeNames)])
        tagsFile.write(tags)   
        if i%1000==0:
            print i
    idsFile.close()
    tagsFile.close()
        

def sampleGen(reader,size):
    j=-1
    for row in reader:
        j+=1
        if j>=size:
            break
        else:
            yield row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10]

def tagsGen(reader,size):
    j=-1
    for row in reader:
        j+=1
        if j>=size:
            break
        else:
            yield row.strip().split(' ')

def idsGen(reader,size):
    j=-1
    for row in reader:
        j+=1
        if j>=size:
            break
        else:
            yield row.strip()




def batchGenerator(batchSize,folderName,fileNamePrefix,totalRows=None):
    if totalRows==None:
        totalRows = getTotalRows('data/'+folderName+'/'+fileNamePrefix+'Ids')
    readerSamples = unicodecsv.reader(open('data/'+folderName+'/'+fileNamePrefix+'Samples.csv', 'r'))   
    readerIds = open('data/'+folderName+'/'+fileNamePrefix+'Ids','r')
    readerTags = open('data/'+folderName+'/'+fileNamePrefix+'Tags','r')
    
    c=0
    while c<totalRows:
        X = sampleGen(readerSamples,batchSize)
        ids = idsGen(readerIds,batchSize)
        Y = list(tagsGen(readerTags,batchSize))
        c += batchSize
        yield ids,X,Y
        
    

def toBinary(tagName,tags):
    Y = []
    for l in tags:
        if tagName in l:
            Y.append(1)
        else:
            Y.append(0)
    return Y


def createTrainTestFiles(inputFileName,folderName,trainSize,testSize,totalSamples=6034196,tags=None):
    if not os.path.exists('data/'+folderName):
        os.makedirs('data/'+folderName)
    writerSampleTrain = unicodecsv.writer(open('data/'+folderName +'/TrainSamples.csv', 'w'))
    writerSampleTest = unicodecsv.writer(open('data/'+folderName +'/TestSamples.csv', 'w'))
    readerSample = unicodecsv.reader(open('data/parsed/'+inputFileName+'Samples.csv', 'r'))
    
    writerIdsTrain = open('data/'+folderName +'/TrainIds', 'w')
    writerIdsTest = open('data/'+folderName +'/TestIds', 'w')
    readerIds = open('data/parsed/'+inputFileName+'Ids', 'r')
    
    writerTagsTrain = open('data/'+folderName +'/TrainTags', 'w')
    writerTagsTest = open('data/'+folderName +'/TestTags', 'w')
    readerTags = open('data/parsed/'+inputFileName+'Tags', 'r')
    

    if trainSize>=totalSamples:
        trainSize = totalSamples-testSize

    i = 0
    trainCount=0
    testCount=0
    for rowSample,rowId,rowTags in izip(readerSample,readerIds,readerTags):
        i+=1
        if tags is not None:
            toContinue = False
            for t in rowTags.split():
                if t not in tags:
                    toContinue = True
                    break
            if toContinue:
                continue
        x = randint(0,totalSamples)
        if x<testSize:
            writerSampleTest.writerow(rowSample)
            writerIdsTest.write(rowId)
            writerTagsTest.write(rowTags)
            testCount+=1
        else:
            if trainCount<trainSize:
                writerSampleTrain.writerow(rowSample)
                writerIdsTrain.write(rowId)
                writerTagsTrain.write(rowTags)
                trainCount+=1
            elif randint(0,5)<2:
                writerSampleTest.writerow(rowSample)
                writerIdsTest.write(rowId)
                writerTagsTest.write(rowTags)
                testCount+=1 
        if testCount>=testSize and trainCount>=trainSize:
            break
    
    print 'train : ' + str(trainCount)
    print 'test : ' + str(testCount)
    print 'total : ' + str(i)




def createRawFile(size,tags):
    reader = unicodecsv.reader(open('data/raw/Train.csv', 'r'))
    outFile = unicodecsv.writer(open('data/raw/TrainSmall.csv','w'))
    i = -1
    c = 0
    for row in reader:
        i+=1
        if i==0:
            outFile.writerow(row)
            continue
        toContinue = False
        for t in row[3].split():
            if t not in tags:
                toContinue = True
                break
        if toContinue:
            continue    

        x = randint(0,20)
        if x<2:
            outFile.writerow(row) 
            c+=1
            if c==size:
                return



def filterTags(tagsFreqTh):
    tags = {}
    filtered = {}
    for line in open('tags', 'r').readlines():
        tup = line.strip().split(' : ')
        freq = int(tup[1])
        if freq<tagsFreqTh:
            filtered[tup[0]] = freq
        else:
            tags[tup[0]] = freq
    return tags,filtered

def getAllTokens(batchSize,folderName,fileNamePrefix,totalRows=None):
    batchGen = batchGenerator(batchSize,folderName,fileNamePrefix,totalRows)
    i = 0
    for _,X,_ in batchGen:
        for sample in X:
            i+=1  
            if i%1000==0:
                print i
            if randint(0,3)<3:
                continue    
            title = sample[0]
            body = sample[1]
            tokenaizedTitle = tokenazie(title,USED_TAGS)
            tokenaizedBody = tokenazie(body,USED_TAGS)
            sents = tokenaizedTitle + tokenaizedBody
            tokens = set()
            for sent in sents:
                for token in sent:
                    tokens.add(token) 
            for token in tokens:
                yield token
                    

def getAllCodeTokens():
    reader = unicodecsv.reader(open('data/raw/Train.csv', 'r'))
    i = -1
    for row in reader:
        i+=1
        if i==0:
            continue  
        if randint(0,3)<3:
            continue   
        if i%1000==0:
            print i 
        if i>1:   
            body = row[2] 
            soup = BeautifulSoup(body)
            codeNodes = soup.findAll('code')
            for codeNode in codeNodes:
                code = u''.join(codeNode.findAll(text=True))
                temp = code.lower().split()
                for w in temp:
                    yield w
                codeNode.extract()


def getAllFormulaTokens():
    formulaRegex = '''\$.+?\$\$?'''
    reader = unicodecsv.reader(open('data/raw/Train.csv', 'r'))
    i = -1
    for row in reader:
        i+=1
        if i==0:
            continue  
        if randint(0,3)<3:
            continue   
        if i%1000==0:
            print i  
        if i>1:   
            body = row[2]
            soup = BeautifulSoup(body)
            codeNodes = soup.findAll('code')
            for codeNode in codeNodes:
                codeNode.extract()
            cleanBody = u''.join(soup.findAll(text=True)).lower()
            matches = re.findall(formulaRegex,cleanBody)
            if matches:
                for m in matches:
                    m = m.replace('$','')
                    temp = m.split()
                    for w in temp:
                        yield w
              
    

def createLexicon(dataFolder):
    fd = FreqDist(getAllTokens(50000,dataFolder,'Train'))
    lex = open('lexicon','w')
    c=0
    for key in fd:
        if fd[key]>=2:
            c+=1
            lex.write(key+'\n')
    lex.close()
    print 'lexicon size : ' + str(c)
    
def createLexiconFormula(maxSamples):
    fd = FreqDist(getAllFormulaTokens())
    lex = open('lexiconFormula','w')
    c=0
    for key in fd:
        if fd[key]>=20:
            c+=1
            lex.write(key.encode('utf-8')+'\n')
    lex.close()
    print 'lexicon size : ' + str(c)
    
def createLexiconCode(maxSamples):
    fd = FreqDist(getAllCodeTokens())
    lex = open('lexiconCode','w')
    c=0
    for key in fd:
        if fd[key]>=100:
            c+=1
            lex.write(key.encode('utf-8')+'\n')
    lex.close()
    print 'lexicon size : ' + str(c)
                    
class Lexicon:
    _end = '_end_'
    def __init__(self,lexiconName):
        self.root = dict()
        lines =  open(lexiconName,'rb').readlines()
        for l in lines:
            word = l.strip()
            current_dict = self.root
            for letter in word:
                current_dict = current_dict.setdefault(letter, {})
            current_dict = current_dict.setdefault(self._end,self._end)

    def contains(self,word):
        current_dict = self.root
        for letter in word:
            if letter in current_dict:
                current_dict = current_dict[letter]
            else:
                return False
        else:
            if self._end in current_dict:
                return True
            else:
                return False

def tokenazie(text,tags):
    sents = SENT_DETECTOR.tokenize(text)
    
    res = [word_tokenize(sent.lower()) for sent in sents]
    for tokens in res:
        i = 0
        while i<len(tokens)-1:
            j=i+1
            if re.search('[^a-z]',tokens[j]):
                temp = tokens[i]+tokens[j]
                if temp in tags:
                    tokens[i] = temp
                    tokens.pop(j)
                    continue
            i+=1
    return res

def getTotalRows(fileName):
    reader1 = open(fileName, 'r')
    totalRows = 0
    for _ in reader1:
        totalRows+=1
    print ('total lines in file {0} : '+str(totalRows)).format(fileName)
    return totalRows
                
class FeatureExtractor:   
    def checkWord(self,word):
        if len(word)>1:
            return word not in self.excludeSet and LEXICON.contains(word)
        if len(word)==1:
            return word in self.includeSet
        return False
    
    def __init__(self):
        self.includeSet = set(['r','c','d'])
        self.excludeSet = set([w for w in STOP_WORDS if w not in ['on','this','where','while']])

    
    def extract(self,sample):
        res = {}
        title = sample[0]
        body = sample[1]
        if sample[2]:
            res['hasCode']=1
            codeNames = sample[10].split()
            for codeName in codeNames:
                res["codeName={}".format(codeName.encode('utf-8'))]=1
        if sample[3]:
            res['hasImage']=1
        if sample[4]:
            res['hasList']=1
        if sample[5]:
            res['hasLink']=1
        if sample[6]:
            res['hasUrl']=1
        if sample[7]:
            res['hasFormula']=1
        formulaTokens = sample[8].split()
        for w in formulaTokens:
            res["formulaToken={}".format(w.encode('utf-8'))]=1
        codeTokens = sample[9].split()
        for w in codeTokens:
            res["codeToken={}".format(w.encode('utf-8'))]=1
         
                               
##        for tag in self.tags:
##            ttag = tag.replace('-',' ')
##            if re.search('(^|\\s){0}(\\s|$|\\.|\\?|!)'.format(re.escape(ttag)),title,re.IGNORECASE):
##                res['tag={}'.format(tag)]=1
##            elif re.search('(^|\\s){0}(\\s|$|\\.|\\?|!)'.format(re.escape(ttag)),body,re.IGNORECASE):
##                res['tag={}'.format(tag)]=1

        tokenaizedTitle = tokenazie(title,USED_TAGS)
        tokenaizedBody = tokenazie(body,USED_TAGS)
        
        bodyTokens = set()
        for tokens in tokenaizedBody:
            for i in range(len(tokens)):
                if tokens[i] not in bodyTokens:
                    if self.checkWord(tokens[i]):
                        bodyTokens.add(tokens[i])                           
        
        for tokens in tokenaizedTitle:
            for i in range(len(tokens)):
                if self.checkWord(tokens[i]) and tokens[i] in bodyTokens:
                    res["tokenBoth={}".format(tokens[i].encode('utf-8'))]=1
        
        sents = tokenaizedTitle + tokenaizedBody
        for tokens in sents:
            for i in range(len(tokens)):
                j=i+1
                if self.checkWord(tokens[i]):
                    res["token={}".format(tokens[i].encode('utf-8'))]=1
                    if j>=len(tokens):
                        continue
                    if  self.checkWord(tokens[j]):
                        res["bigram={} {}".format(tokens[i].encode('utf-8'),tokens[j].encode('utf-8'))]=1       
        return res       

def trainClassifier(batchSize,dataFolder,clfFolderName,tagsSplitSize):
    startTime = time()
    if not os.path.exists(clfFolderName):
        os.makedirs(clfFolderName)
    if not os.path.exists(clfFolderName+'Temp'):
        os.makedirs(clfFolderName+'Temp')
    tags = list(USED_TAGS.keys())
    totalRows = getTotalRows('data/'+dataFolder+'/TrainIds')
     
    hasher = FeatureHasher()
    batchGen = batchGenerator(batchSize,dataFolder,'Train',totalRows)   
    hashInd = 1
    print 'number of tags : ' + str(len(tags))
    extractor = FeatureExtractor()
    for _,X,_ in batchGen:
        batchTime = time()
        print 'computing batch : ' + str(hashInd)
        X_batch = hasher.transform(extractor.extract(sample) for sample in X)
        print 'saving batch : ' + str(hashInd)
        with open(clfFolderName+'Temp/'+str(hashInd)+'.pkl', 'wb') as fid:
                cPickle.dump(X_batch, fid)
        print 'batch time : ' + str(time()-batchTime)
        hashInd+=1
    with open(clfFolderName+'/hasher.pkl', 'wb') as fid:
        cPickle.dump(hasher, fid)
    with open(clfFolderName+'/extractor.pkl', 'wb') as fid:
        cPickle.dump(extractor, fid)
    print 'hashing time : ' + str(time()-startTime)
    
    tagIndDic = {}
    tagInd = 1 
    loop = 1
    for currTags in [tags[i:i+tagsSplitSize] for i in range(0,len(tags),tagsSplitSize)]:
        iterStartTime = time()
        print 'tags iteration : ' + str(loop)
        clfDic = {}
        for tag in currTags:
            clfDic[tag] = Perceptron(alpha=ALPHA,n_iter=N_ITER)
        batchGen = batchGenerator(batchSize,dataFolder,'Train',totalRows)
        batchInd = 1
        for _,_,targets_in_batch in batchGen:
            batchTime = time()
            print 'batch number : ' + str(batchInd)
            with open(clfFolderName+'Temp/'+str(batchInd)+'.pkl','rb') as fp:
                X_batch=cPickle.load(fp)
            for tag in currTags:
                Y_batch_binary = toBinary(tag,targets_in_batch)
                clfDic[tag].partial_fit(X_batch, Y_batch_binary, classes=[0,1])
            batchInd+=1
            print 'batch time : ' + str(time()-batchTime)
        for tag in clfDic:
            clfDic[tag].sparsify()
            tagIndDic[tag]=tagInd
            with open(clfFolderName+'/'+str(tagInd)+'.pkl', 'wb') as fid:
                cPickle.dump(clfDic[tag], fid)
            tagInd+=1
        loop+=1
        print 'iter time : ' + str(time()-iterStartTime)
        print 
    print 'saving model...'
    with open(clfFolderName+'/tagIndDic.pkl', 'wb') as fid:
        cPickle.dump(tagIndDic, fid)

    print 'total time : ' + str(time()-startTime)
     
     
class Classifier:
    def __init__(self,folder):
        self.folder=folder + '/'
        with open(self.folder + '/tagIndDic.pkl','rb') as fp:
            self.tagIndDic=cPickle.load(fp)
        with open(self.folder + '/hasher.pkl','rb') as fp:
            self.hasher=cPickle.load(fp)
        with open(self.folder + '/extractor.pkl','rb') as fp:
            self.extractor=cPickle.load(fp)

    def predict(self,X):
        res = []
        i=1
        X_trans = self.hasher.transform(self.extractor.extract(sample) for sample in X)
        print 'done hashing'
        startTime = time()
        for tag in self.tagIndDic:
            if i % 100==0:
                print str(i) + ' binary time : ' + str(time()-startTime)
                startTime = time()
            tagInd = self.tagIndDic[tag]
            clf = None
            with open(self.folder+str(tagInd)+'.pkl','rb') as fp:
                clf=cPickle.load(fp)
            pred =  clf.predict(X_trans)
            if len(res)==0:
                res = [[] for _ in pred]
            for p,l in izip(pred,res):
                if p==1:
                    l.append(tag)
            i+=1 
        return res


    def decision_function(self,X,th=0):
        res = []
        i=1
        X_trans = self.hasher.transform(self.extractor.extract(sample) for sample in X)
        print 'done hashing'
        startTime = time()
        for tag in self.tagIndDic:
            if i % 100==0:
                print str(i) + ' binary time : ' + str(time()-startTime)
                startTime = time()
            tagInd = self.tagIndDic[tag]
            clf = None
            with open(self.folder+str(tagInd)+'.pkl','rb') as fp:
                clf=cPickle.load(fp)
            pred =  clf.decision_function(X_trans)
            if len(res)==0:
                res = [[] for _ in pred]
            for p,l in izip(pred,res):
                if p>th:
                    l.append((tag,p))
            i+=1
        return res
    
    def predictTh(self,X,th):
        temp = self.decision_function(X,th)
        res = []
        for l in temp:
            if len(l)>5:
                l.sort(key=lambda tup: -1 * tup[1])
                l = l[:5]
            pred = [t for t,v in l if v>th] 
            if len(pred)==0 and len(l)>0:
                pred = [l[0][0]]
            res.append(pred)
        return res
    
    def predictTagsWhiteList(self,X,tagsWhiteList):
        dec = self.decision_function(X)
        res = []
        for l in dec:
            l.sort(key=lambda tup: -1 * tup[1])
            if len(l)==0:
                res.append([])
                continue
            temp = []
            first = True
            for t,p in l:
                if first:
                    first=False
                    temp.append(t)
                    continue
                toAdd = True
                for t2 in temp:
                    if t not in tagsWhiteList[t2]:
                        toAdd = False
                if toAdd:
                    temp.append(t)
            res.append(temp)
        return res
                
                
        

def predCountFreq(predictions):
    d = {}
    for p in predictions:
        d[len(p)] = d.get(len(p),0)+1
    return d

def prepareForSub(classifierFolderName,batchSize):
    writer  = unicodecsv.writer(open('final.csv', 'w'))
    writer.writerow(['Id','Tags'])
    clf = Classifier(classifierFolderName)
    batchGen = batchGenerator(batchSize,'parsed','Test')
    i = 1
    for ids,X,_ in batchGen:
        startTime = time()
        print 'batch : ' + str(i)
        pred = clf.predict(X)
        for idd,p in izip(ids,pred):
            writer.writerow([idd,' '.join(p)])
        i+=1
        print 'batch time : ' + str(time()-startTime)
    print i

def evaluateTag(pred,exp,tag):
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    totalP=0
    totalE=0
    for p,e in izip(pred,exp):
        inP = tag in p
        if inP:
            totalP+=1
            
        inE = tag in e
        if inE:
            totalE+=1
            
        if inP and inE:
            tp+=1
        if inP and not inE:
            fp+=1
        if not inP and inE:
            fn+=1
        if not inP and not inE:
            tn+=1
    if tp == 0:
        precision=0
        recall = 0
        f_score = 0
    else:
        precision = tp / (tp + fp)
        recall =  tp / (tp + fn)
        f_score = (2*precision*recall)/(precision+recall)
    return precision,recall,f_score,totalP,totalE



def evaluateClassifier(testFolderName,clfFolderName,th=0):
    clf = Classifier(clfFolderName)
    batchGen = batchGenerator(50000,testFolderName,'Test')
    pred = []
    exp = []
    for _,X,Y in batchGen:
        pred += clf.predict(X)
        exp += Y
    print precision_recall_fscore_support(exp, pred, average='micro')
    return pred,exp

def evaluateTags(clfFolderName,testFolderName):
    output = open('tagsEvaluation.csv','w')
    pred,exp = evaluateClassifier(testFolderName,clfFolderName)
    output.write('tag,freq, precision,recall,f_score,totalP,totalE\n')
    for tag in USED_TAGS.keys():
        precision,recall,f_score,totalP,totalE = evaluateTag(pred, exp, tag)
        output.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(tag,USED_TAGS[tag],precision,recall,f_score,totalP,totalE))
    output.close()

def createTagsWhiteList(tags,batchSize,dataFolder,totalRows=None): 
    batchGen = batchGenerator(batchSize,dataFolder,'Train',totalRows)  
    d = {}
    for t in tags:
        d[t] = set()
    i = 0
    for _,_,Y in batchGen:
        for y in Y:
            i+=1
            if i%100000==0:
                print i
            for y1 in y:
                if y1 in tags:
                    for y2 in y:
                        if y1!=y2:
                            d[y1].add(y2)
    return d


def findBestTh(clfFolderName,testFolderName):
    clf = Classifier(clfFolderName)
    batchGen = batchGenerator(50000,testFolderName,'Test')
    predTuple = []
    exp = []
    for _,X,Y in batchGen:
        predTuple += clf.decision_function(X)
        exp += Y
    
    for i in range(20):
        th = 0.2*i
        temp = predTuple
        res = []
        for l in temp:
            if len(l)>5:
                l.sort(key=lambda tup: -1 * tup[1])
                l = l[:5]
            pred = [t for t,v in l if v>th]
            if len(pred)==0 and len(l)>0:
                pred = [l[0][0]]
            res.append(pred)
        print th
        print precision_recall_fscore_support(exp, res, average='micro')
        print



#createLexiconCode(10000000)
#createLexiconFormula(10000000)
####### STATIC FIELDS ########
USED_TAGS,FILTERED_TAGS = filterTags(50)
SENT_DETECTOR = data.load('tokenizers/punkt/english.pickle')
STOP_WORDS = createStopWords()
LEXICON_CODE = Lexicon('lexiconCode')
LEXICON_FORMULA = Lexicon('lexiconFormula')
LEXICON = Lexicon('lexicon')
ALPHA = 0.0001
N_ITER = 50
###############################




#parseHtmlSamples('TrainSmall')
#parseHtmlSamples('Test')
#createTrainTestFiles('TrainSmall','100k',100000,20000,120000,USED_TAGS)
trainClassifier(50000,'large','clf',150)
evaluateClassifier('large','clf')
#prepareForSub('clf3',505000)
#evaluateTags('clf','large')
