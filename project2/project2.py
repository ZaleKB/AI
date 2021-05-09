# -------------------------------------------------------
# Project (include number)
# Written by (include your name and student id)
# For COMP 6721 Section (your lab section) – Fall 2019
# --------------------------------------------------------

import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tree import Tree
import math
import time
import sys
import os
import docx
import matplotlib.pyplot as plt
import operator


lemmatizer = WordNetLemmatizer()


contain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
       'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
       'x', 'y', 'z']  #if word not contain any number and character, delete

starts = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
       'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
       'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9') #the word start can only be with these

space = ('a ', 'b ', 'c ', 'd ', 'e ', 'f ', 'g ', 'h ', 'i ',
       'j ', 'k ', 'l ', 'm ', 'n ', 'o ', 'p ', 'q ', 'r ', 's ', 't ', 'u ', 'v ', 'w ',
       'x ', 'y ', 'z ', 'car ', 'c/', 'c.', 'b.','b-', 'a.', 'a-', 'c-', 'd-', 'd.',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'f-', 'f*', 'f.') #if start with these, delete the word

npp = None
story = None   #numpy to classifed each words inside
ask_hn = None
show_hn = None
poll = None
process = None
testing = None

P_story = None # P(story) etc
P_ask = None
P_show = None
P_poll =None

smoothed = 0.5  # smoothed number

data_model = None

stopWordsRun = False
stopwords = [] #stopwords list task 1.3.1
word_length_filter = False #task 1.3.2

freq1 = False   #for task 1.3.3
freq5 = False
freq10 = False
freq15 = False
freq20 = False

top5 = False  #for task 1.3.3 part2
top10 = False
top15 = False
top20 = False
top25 = False
test = None


def separate_file():
    global testing
    global test
    global story
    global ask_hn
    global show_hn
    global poll
    global process
    global np_T
    
    
    global P_story
    global P_ask
    global P_show
    global P_poll
    
    df1 = pd.read_csv(r'E:\COMP 6721\hn2018_2019.csv')  # read file
    
    training = df1.loc[df1['Created At'].str.startswith('2018')]  # training dataset 2018 years
    testing = df1.loc[df1['Created At'].str.startswith('2019')]
    
    #total dataset
    process = training[['Title', 'Post Type']]
    #df3 = process.copy()
    #np_P = np.array(df3)
    
    # testing dataset
    test = testing[['Title', 'Post Type']]#.iloc[0:600]
    np_T = np.array(test['Title'].copy())

    # separate into 4 class, and handle them alone
    story = process.loc[process['Post Type'] == 'story']#.iloc[0:3000].copy()
    story = np.array(story['Title'])
    ask_hn = process.loc[process['Post Type'] == 'ask_hn']#.iloc[0:3000].copy()
    ask_hn = np.array(ask_hn['Title'])
    show_hn = process.loc[process['Post Type'] == 'show_hn']#.iloc[0:3000].copy()
    show_hn = np.array(show_hn['Title'])
    poll = process.loc[process['Post Type'] == 'poll']#.iloc[0:3000].copy()
    poll = np.array(poll['Title'])
    
    
    total_doc = len(story) + len(ask_hn) + len(show_hn) + len(poll)
    
    P_story = math.log10(len(story)/total_doc)
    P_ask = math.log10(len(ask_hn)/total_doc)   #要转移
    P_show = math.log10(len(show_hn)/total_doc)
    P_poll = math.log10(len(poll)/total_doc)
    
    
    
    

# 得到词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V') or tag.startswith('I') or tag.startswith('PR'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# tokenize and rule grammar 
# 
# 
def preprocess(text):
    sentence = text.lower()

    sentence = word_tokenize(sentence)
    sentence = pos_tag(sentence)

    word_pos_list = []
    grammar = r"""
    NP: {<NNS|NN><NN|NNS>}
    NP_P:  {<NN|NNS><POS>}
    """
    cp = nltk.RegexpParser(grammar)
    cs = cp.parse(sentence)

    for i in cs:
        if type(i) != Tree:
            word_pos_list.append(i)
        elif type(i) == Tree:
            # print(i)
            if i.label() == "NP":
                phrase = (i[0][0] + " " + i[1][0], i.label())
                word_pos_list.append(phrase)
            elif i.label() == "NP_P":
                phrase = (i[0][0] + i[1][0], i.label())
                word_pos_list.append(phrase)

    return word_pos_list


meanningless = []  # meaningless vocabulary

# lemmatize each sentence
def lemmatize_sentence(sentence):
    global meanningless
    meanningless = []
    
    result = []
    for word, pos in preprocess(sentence):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        each = lemmatizer.lemmatize(word, pos=wordnet_pos)
        if each.isdigit() or any(word in each for word in contain) == False or each.startswith(starts) == False or each.startswith(space):
            continue
        else:
            result.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return result
  


dirForTotal = {}
dirStory = {}
dirAsk = {}
dirShow = {}
dirPoll = {}


def after_lemmatize_training():
    
    global story
    global ask_hn
    global show_hn
    global poll
     
    start = time.clock()
    
    # 相当于把整个文本分成4段处理
    for sentence in range(len(story)):
        eachSentence = lemmatize_sentence(story[sentence])
        story[sentence] = eachSentence
        
    story = [x for x in story if x != []]
    story = np.asarray(story)  # delete empty text line
        
    for sentence in range(len(ask_hn)):
        eachSentence = lemmatize_sentence(ask_hn[sentence])
        ask_hn[sentence] = eachSentence
        
    ask_hn = [x for x in ask_hn if x != []]
    ask_hn = np.asarray(ask_hn)

    for sentence in range(len(show_hn)):
        eachSentence = lemmatize_sentence(show_hn[sentence])
        show_hn[sentence] = eachSentence
        
    show_hn = [x for x in show_hn if x != []]
    show_hn = np.asarray(show_hn)
        
    for sentence in range(len(poll)):
        eachSentence = lemmatize_sentence(poll[sentence])
        poll[sentence] = eachSentence
        
    poll = [x for x in poll if x != []]
    poll = np.asarray(poll)
    
    
    elapsed = (time.clock() - start)
    print('Time used to lemmatized is:', elapsed)
        
    
    # 后续处理，包括删除一些无意义的词

def after_lemmatize_testing():
    start = time.clock()
    for sentence in range(len(np_T)):
        eachSentence = lemmatize_sentence(np_T[sentence])
        np_T[sentence] = eachSentence
    
    elapsed = (time.clock() - start)
    print('Time used to test_lemmatize is:', elapsed)



def classified():
    separate_file()
    after_lemmatize_training()
    after_lemmatize_testing()
    
    start = time.clock()
    for sentence in range(len(story)):  #build the dictionary for story
        wordlist = story[sentence]
        for word in wordlist:
            if word in dirStory:
                dirStory[word] += 1
            if word not in dirStory:
                dirStory[word] = 1
            if word in dirForTotal:
                dirForTotal[word] += 1
            if word not in dirForTotal:
                dirForTotal[word] = 1 
            
                         
    for sentence in range(len(ask_hn)):  #build the dictionary for ask_hn
        wordlist = ask_hn[sentence]
        for word in wordlist:           
            if word in dirAsk:
                dirAsk[word] += 1
            if word not in dirAsk:
                dirAsk[word] = 1
            if word in dirForTotal:
                dirForTotal[word] += 1
            if word not in dirForTotal:
                dirForTotal[word] = 1
            
    
    for sentence in range(len(show_hn)):  #build the dictionary for show_hn
        wordlist = show_hn[sentence]
        for word in wordlist:            
            if word in dirShow:
                dirShow[word] += 1
            if word not in dirShow:
                dirShow[word] = 1
            if word in dirForTotal:
                dirForTotal[word] += 1
            if word not in dirForTotal:
                dirForTotal[word] = 1                        
            
                
    for sentence in range(len(poll)):  #build the dictionary for poll
        wordlist = poll[sentence]
        for word in wordlist:
            if word in dirPoll:
                dirPoll[word] += 1
            if word not in dirPoll:
                dirPoll[word] = 1
            if word in dirForTotal:
                dirForTotal[word] += 1
            if word not in dirForTotal:
                dirForTotal[word] = 1
                        
        
    elapsed = (time.clock() - start)
    print('Time used to classified is:', elapsed)
 
    
def print_vocabulary(table):
    vocabulary_table = open(table, 'w+', encoding="utf-8")
    
    for word in dirForTotal:
        vocabulary_table.write(word+'  '+'\n')
        vocabulary_table.flush()
    vocabulary_table.close()



voca = None
storyP = None
AskP = None
ShowP = None
PollP = None

top5story = None
top5ask = None
top5show = None
top5poll = None
top5total = None

top10story = None
top10ask = None
top10show = None
top10poll = None
top10total = None

top15story = None
top15ask = None
top15show = None
top15poll = None
top15total = None

top20story = None
top20ask = None
top20show = None
top20poll = None
top20total = None

top25story = None
top25ask = None
top25show = None
top25poll = None
top25total = None

dirForIndexing = {}

def create_file(text):
    
    global data_model
    
    global voca
    global storyP
    global AskP
    global ShowP
    global PollP
    global dirForIndexing
    
    global top5story
    global top5ask
    global top5show
    global top5poll
    global top5total
    
    global top10story
    global top10ask
    global top10show
    global top10poll
    global top10total
    
    global top15story
    global top15ask
    global top15show
    global top15poll
    global top15total
    
    global top20story
    global top20ask
    global top20show
    global top20poll
    global top20total
    
    global top25story
    global top25ask
    global top25show
    global top25poll
    global top25total
    
    
    start = time.clock()
    
    total_file = dirForTotal.copy()      #keep the original dictionary and make copy for all dictionary
    story_file = dirStory.copy()
    ask_file = dirAsk.copy()
    show_file = dirShow.copy()
    poll_file = dirPoll.copy()
    
    
    temp_total = total_file.copy()
    
    sorted_key = sorted(dirForTotal.items(),key=operator.itemgetter(1),reverse=True)
    d = dict(sorted_key)
    
    
    if word_length_filter:
        for word in temp_total:
            if len(word) <= 2 or len(word) >= 9:
                del total_file[word]
                if word in story_file:
                    del story_file[word]
                if word in ask_file:
                    del ask_file[word]
                if word in show_file:
                    del show_file[word]
                if word in poll_file:
                    del poll_file[word]
    
    if stopWordsRun:
        for word in temp_total:
            if word in stopwords:
                del total_file[word]
                if word in story_file:
                    del story_file[word]
                if word in ask_file:
                    del ask_file[word]
                if word in show_file:
                    del show_file[word]
                if word in poll_file:
                    del poll_file[word]
            
            
    if freq1:
        for word in temp_total:
            if total_file.get(word) == 1:
                del total_file[word]
            if story_file.get(word) == 1:
                del story_file[word]
            if ask_file.get(word) == 1:
                del ask_file[word]
            if show_file.get(word) == 1:
                del show_file[word]
            if poll_file.get(word) == 1:
                del poll_file[word]
                
    if freq5:
        for word in temp_total:
            if total_file.get(word) <= 5:
                del total_file[word]
            if story_file.get(word) != None and story_file[word] <= 5:
                del story_file[word]
            if ask_file.get(word) != None and ask_file[word] <= 5:
                del ask_file[word]
            if show_file.get(word) != None and show_file[word] <= 5:
                del show_file[word]
            if poll_file.get(word) != None and poll_file[word] <= 5:
                del poll_file[word]
    
    if freq10:
        for word in temp_total:
            if total_file.get(word) <= 10:
                del total_file[word]
            if story_file.get(word) != None and story_file[word] <= 10:
                del story_file[word]
            if ask_file.get(word) != None and ask_file[word] <= 10:
                del ask_file[word]
            if show_file.get(word) != None and show_file[word] <= 10:
                del show_file[word]
            if poll_file.get(word) != None and poll_file[word] <= 10:
                del poll_file[word]
                
    if freq15:
        for word in temp_total:
            if total_file.get(word) <= 15:
                del total_file[word]
            if story_file.get(word) != None and story_file[word] <= 15:
                del story_file[word]
            if ask_file.get(word) != None and ask_file[word] <= 15:
                del ask_file[word]
            if show_file.get(word) != None and show_file[word] <= 15:
                del show_file[word]
            if poll_file.get(word) != None and poll_file[word] <= 15:
                del poll_file[word]
                
    if freq20:
        for word in temp_total:
            if total_file.get(word) <= 20:
                del total_file[word]
            if story_file.get(word) != None and story_file[word] <= 20:
                del story_file[word]
            if ask_file.get(word) != None and ask_file[word] <= 20:
                del ask_file[word]
            if show_file.get(word) != None and show_file[word] <= 20:
                del show_file[word]
            if poll_file.get(word) != None and poll_file[word] <= 20:
                del poll_file[word]
                
    if top5:
        cd5 = d.copy()
        top = int(len(cd5) * 0.05)
        delete = {k: cd5[k] for k in list(cd5)[:top]}
        
        for word in delete:
            del total_file[word]
            if word in story_file:
                del story_file[word]
            if word in ask_file:
                del ask_file[word]
            if word in show_file:
                del show_file[word]
            if word in poll_file:
                del poll_file[word]
        
        top5story = story_file
        top5ask = ask_file
        top5show = show_file
        top5poll = poll_file
        top5total = total_file
        
        
    if top10:
        sorted_key = sorted(top5total.items(),key=operator.itemgetter(1),reverse=True)
        d = dict(sorted_key)
        cd10 = d.copy()
        top = int(len(cd10) * 0.1)
        delete = {k: cd10[k] for k in list(cd10)[:top]}
        
        top10story = top5story.copy()
        top10ask = top5ask.copy()
        top10show = top5show.copy()
        top10poll = top5poll.copy()
        top10total = top5total.copy()
        
        for word in delete:
            del top10total[word]
            if word in top10story:
                del top10story[word]
            if word in top10ask:
                del top10ask[word]
            if word in top10show:
                del top10show[word]
            if word in top10poll:
                del top10poll[word]
        
        story_file = top10story
        ask_file = top10ask
        show_file = top10show
        poll_file = top10poll
        total_file = top10total
        
    if top15:
        sorted_key = sorted(top10total.items(),key=operator.itemgetter(1),reverse=True)
        d = dict(sorted_key)
        cd15 = d.copy()
        top = int(len(cd15) * 0.1)
        delete = {k: cd15[k] for k in list(cd15)[:top]}
        
        top15story = top10story.copy()
        top15ask = top10ask.copy()
        top15show = top10show.copy()
        top15poll = top10poll.copy()
        top15total = top10total.copy()
        
        for word in delete:
            del top15total[word]
            if word in top15story:
                del top15story[word]
            if word in top15ask:
                del top15ask[word]
            if word in top15show:
                del top15show[word]
            if word in top15poll:
                del top15poll[word]
        
        story_file = top15story
        ask_file = top15ask
        show_file = top15show
        poll_file = top15poll
        total_file = top15total
        
    if top20:
        sorted_key = sorted(top15total.items(),key=operator.itemgetter(1),reverse=True)
        d = dict(sorted_key)
        cd20 = d.copy()
        top = int(len(cd20) * 0.1)
        delete = {k: cd20[k] for k in list(cd20)[:top]}
        
        top20story = top15story.copy()
        top20ask = top15ask.copy()
        top20show = top15show.copy()
        top20poll = top15poll.copy()
        top20total = top15total.copy()
        
        for word in delete:
            del top20total[word]
            if word in top20story:
                del top20story[word]
            if word in top20ask:
                del top20ask[word]
            if word in top20show:
                del top20show[word]
            if word in top20poll:
                del top20poll[word]
        
        story_file = top20story
        ask_file = top20ask
        show_file = top20show
        poll_file = top20poll
        total_file = top20total
        
    if top25:
        sorted_key = sorted(top20total.items(),key=operator.itemgetter(1),reverse=True)
        d = dict(sorted_key)
        cd25 = d.copy()
        top = int(len(cd25) * 0.1)
        delete = {k: cd25[k] for k in list(cd25)[:top]}
        
        top25story = top20story.copy()
        top25ask = top20ask.copy()
        top25show = top20show.copy()
        top25poll = top20poll.copy()
        top25total = top20total.copy()
        
        for word in delete:
            del top25total[word]
            if word in top25story:
                del top25story[word]
            if word in top25ask:
                del top25ask[word]
            if word in top25show:
                del top25show[word]
            if word in top25poll:
                del top25poll[word]
        
        story_file = top25story
        ask_file = top25ask
        show_file = top25show
        poll_file = top25poll
        total_file = top25total
        


    VStory = len(story_file)
    VAsk = len(ask_file)
    VShow = len(show_file)
    VPoll = len(poll_file)
    Vword = len(total_file)
    
    
    temp_story = story_file.copy() #assign the frequency
    temp_ask = ask_file.copy()
    temp_show = show_file.copy()
    temp_poll = poll_file.copy()
    
    key = list(total_file.keys())
    list.sort(key)
    #keys = key
    

    file = open(text, "w+", encoding="utf-8")
    
    #file_for_me = open('analyse.txt', 'w+', encoding='utf-8')

    for i in range(len(key)):
        file.write(str(i + 1)+'  '+str(key[i])+'  ' +
                   str(temp_story.setdefault(key[i], 0)) + '  ' + str(round(math.log10((smoothed + temp_story.setdefault(key[i], 0))/(Vword + VStory)), 10))+'  ' +
                   str(temp_ask.setdefault(key[i], 0)) + '  ' + str(round(math.log10((smoothed + temp_ask.setdefault(key[i], 0))/(Vword + VAsk)), 10))+'  ' +
                   str(temp_show.setdefault(key[i], 0)) + '  ' + str(round(math.log10((smoothed + temp_show.setdefault(key[i], 0))/(Vword + VShow)), 10))+'  ' +
                   str(temp_poll.setdefault(key[i], 0)) + '  ' + str(round(math.log10((smoothed + temp_poll.setdefault(key[i], 0))/(Vword + VPoll)), 10))+'  ' +
                   '\n')
        file.flush()
    file.close()
    
    data = pd.read_csv(text, header=None, sep = '  ' , keep_default_na=False, na_values=['NA'])
    data.columns=['serial', 'word', 'StoryF', 'StoryP','AskF','AskP', 'ShowF', 'ShowP', 'PollF', 'PollP']
    data_model = data
    
    voca = data_model['word'].values 
    storyP = data_model['StoryP'].values  # build words location for each class to check later
    AskP = data_model['AskP'].values
    ShowP = data_model['ShowP'].values
    PollP = data_model['PollP'].values
    
    for i in range(len(voca)):
        dirForIndexing[voca[i]] = i   # build the dictionary for indexing all the word in dictionary
        
        
    elapsed = (time.clock() - start)
    print('Time used to create file is:', elapsed)



def naiive_bayes(text, title):
    create_file(text)
    
    start = time.clock()
    
    np_test = test[['Title', 'Post Type']].values
    
    c_class = None
    result = None
    
    score_Story = P_story
    score_Ask = P_ask
    score_Show = P_show
    score_Poll = P_poll
    
    file = open(title, 'w+', encoding='utf-8')
    file2 = open('use_' + title, 'w+', encoding='utf-8')  #file2 is for me to use sep with ,,, easy to open with no mistake
    for line in range(len(np_T)):
        wordlist = np_T[line]
        
        for word in wordlist:
            if word in dirForIndexing:
                
                score_Story +=  storyP[dirForIndexing[word]]
                score_Ask += AskP[dirForIndexing[word]]
                score_Show += ShowP[dirForIndexing[word]]
                score_Poll += PollP[dirForIndexing[word]]
                
            else:
                continue
        
        max_value = max(score_Story, score_Ask, score_Show, score_Poll)
        
        if max_value is score_Story:
            c_class = 'story'
        elif max_value is score_Ask:
            c_class = 'ask_hn'
        elif max_value is score_Show:
            c_class = 'show_hn'
        else:
            c_class = 'poll'
            
        if c_class == np_test[line][1]:
            result = 'right'
        else:
            result = 'wrong'
            
        
        file2.write(str(line + 1)+',,,'+str(np_test[line][0])+',,,'+str(np_test[line][1])+',,,'+
            str(score_Story)+',,,'+str(score_Ask)+',,,'+str(score_Show)+',,,'+str(score_Poll)+',,,'+
            c_class+',,,'+result+'\n')
        
        file2.flush()
        
        file.write(str(line + 1)+'  '+str(np_test[line][0])+'  '+str(np_test[line][1])+'  '+
            str(score_Story)+'  '+str(score_Ask)+'  '+str(score_Show)+'  '+str(score_Poll)+'  '+
            c_class+'  '+result+'\n')
            
        file.flush()
        
        score_Story = P_story
        score_Ask = P_ask
        score_Show = P_show
        score_Poll = P_poll
    
    file.close()
    file2.close()
    
    elapsed = (time.clock() - start)
    print('Time used to testing is:', elapsed)



def calculate_value(text):  #caculate four values accuracy, precision, recall, F1-measure
    num = 0
    for i in range(len(text)):
        if text[i][2] == 'wrong':
            num += 1


    accuracy = (len(text) - num)/len(text)
    
    story_num = 0; story_wrong = 0
    ask_num = 0; ask_wrong = 0
    show_num = 0; show_wrong = 0
    poll_num = 0; poll_wrong = 0
    
    
    for i in range(len(text)):
        if text[i][0] == 'story':
            story_num += 1
            if text[i][2] == 'wrong':
                story_wrong += 1
        
        elif text[i][0] == 'ask_hn':
            ask_num += 1
            if text[i][2] == 'wrong':
               ask_wrong += 1
          
        elif text[i][0] == 'show_hn':
            show_num += 1
            if text[i][2] == 'wrong':
               show_wrong += 1
               
        elif text[i][0] == 'poll':
            poll_num += 1
            if text[i][2] == 'wrong':
               poll_wrong += 1
               
    
    
    if story_num is 0:
        recall_story = -1
    else:
        recall_story = (story_num - story_wrong)/story_num
    
    if ask_num is 0:
        recall_ask = -1
    else:
        recall_ask = (ask_num - ask_wrong)/ask_num 
        
    if show_hn is 0:
        recall_show = -1
    else:
        recall_show = (show_num - show_wrong)/show_num
        
    if poll_num is 0:
        recall_poll = -1
    else:
        recall_poll = (poll_num - poll_wrong)/poll_num
        
    
    story_pre = 0; story_right = 0
    ask_pre = 0; ask_right = 0
    show_pre = 0; show_right = 0
    poll_pre = 0; poll_right = 0
    
    for i in range(len(text)):
        if text[i][1] == 'story':
            story_pre += 1
            if text[i][2] == 'right':
                story_right += 1
                
        elif text[i][1] == 'ask_hn':
            ask_pre += 1
            if text[i][2] == 'right':
                ask_right += 1
                
        elif text[i][1] == 'show_hn':
            show_pre += 1
            if text[i][2] == 'right':
                show_right += 1 
                
        elif text[i][1] == 'poll':
            poll_pre += 1
            if text[i][2] == 'right':
                poll_right += 1
            

    if story_pre is 0:
        precision_story = -1
    else:
        precision_story = story_right/story_pre
        
    if ask_pre is 0:
        precision_ask = -1
    else:
        precision_ask = ask_right/ask_pre
        
    if show_pre is 0:
        precision_show = -1
    else:
        precision_show = show_right/show_pre
        
    if poll_pre is 0:
        precision_poll = -1
    else:
        precision_poll = poll_right/poll_pre
        
    
    if (precision_story + recall_story) <= 0:
        F1_story = -1
    else:
        F1_story = (2 * precision_story * recall_story)/(precision_story + recall_story)
        
    if (precision_ask + recall_ask) <= 0:
        F1_ask = -1
    else:
        F1_ask = (2 * precision_ask * recall_ask)/(precision_ask + recall_ask)
        
    if (precision_show + recall_show) <= 0:
        F1_show = -1
    else:
        F1_show = (2 * precision_show * recall_show)/(precision_show + recall_show)
        
    if (precision_poll + recall_poll) <= 0:
        F1_poll = -1
    else:
        F1_poll = (2 * precision_poll * recall_poll)/(precision_poll + recall_poll)
        
    
    #build a dictionary for computation
    values = {'accuracy': accuracy, 'recall_story': recall_story, 'recall_ask': recall_ask, 'recall_show': recall_show, 'recall_poll': recall_poll,
              'precision_story': precision_story, 'precision_ask': precision_ask, 'precision_show': precision_show, 'precision_poll': precision_poll,
              'F1_story': F1_story, 'F1_ask': F1_ask, 'F1_show': F1_show, 'F1_poll': F1_poll}
    
    return values
    
    

def generate_matrix(text):  # build confusion matrix for task1
    
    matrix = np.zeros((4,5), dtype = np.int64)
    
  
    for i in range(len(text)):
        if text[i][0] == 'story':
            matrix[0][4] += 1
            
            if text[i][1] == 'story':
                matrix[0][0] += 1
                
            elif text[i][1] == 'ask_hn':
                matrix[0][1] += 1
                
            elif text[i][1] == 'show_hn':
                matrix[0][2] += 1
            
            elif text[i][1] == 'poll':
                matrix[0][3] += 1
                
            
        if text[i][0] == 'ask_hn':
            matrix[1][4] += 1
            
            if text[i][1] == 'story':
                matrix[1][0] += 1
                
            elif text[i][1] == 'ask_hn':
                matrix[1][1] += 1
                
            elif text[i][1] == 'show_hn':
                matrix[1][2] += 1
            
            elif text[i][1] == 'poll':
                matrix[1][3] += 1
        
        
        if text[i][0] == 'show_hn':
            matrix[2][4] += 1
            
            if text[i][1] == 'story':
                matrix[2][0] += 1
                
            elif text[i][1] == 'ask_hn':
                matrix[2][1] += 1
                
            elif text[i][1] == 'show_hn':
                matrix[2][2] += 1
            
            elif text[i][1] == 'poll':
                matrix[2][3] += 1
        
        if text[i][0] == 'poll':
            matrix[3][4] += 1
            
            if text[i][1] == 'story':
                matrix[3][0] += 1
                
            elif text[i][1] == 'ask_hn':
                matrix[3][1] += 1
                
            elif text[i][1] == 'show_hn':
                matrix[3][2] += 1
            
            elif text[i][1] == 'poll':
                matrix[3][3] += 1
                
    return matrix
            
            
            
        

basvalue = None
basMatrix = None

def base_model():
    global stopwords
    global stopWordsRun
    global basvalue
    global dirForIndexing
    global basMatrix
    
    dirForIndexing = {} #clear the dict every time
    stopWordsRun = False
    stopwords = []
    classified()
    
    print_vocabulary('vocabulary.txt')
    naiive_bayes('model-2018.txt', 'baseline-result.txt')
    
    bas = pd.read_csv('use_baseline-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    bas.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    
    Matrix = np.array(bas[['type', 'predic']])
    
    bas = np.array(bas[['type', 'predic', 'result']])
    
    
    basvalue = calculate_value(bas)
    
    basMatrix = generate_matrix(Matrix)

    
stopvalue = None   
stopmatrix = None 
  
def stopword_filtering():
    global stopwords
    global stopWordsRun
    global stopvalue
    global dirForIndexing
    global stopmatrix
    
    dirForIndexing = {}
    
    file=docx.Document("E:\COMP 6721\Stopwords.docx")
    
    for para in file.paragraphs:
        if para.text != '':
            stopwords.append(para.text)
    
    
    stopWordsRun = True    
    #classified()
    naiive_bayes('stopword-model.txt', 'stopword-result.txt')
    
    stop = pd.read_csv('use_stopword-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    stop.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    
    Matrix = np.array(stop[['type', 'predic']])
    stop = np.array(stop[['type', 'predic', 'result']])
    
    stopvalue = calculate_value(stop)
    
    stopmatrix = generate_matrix(Matrix)

    
worvalue = None
lenmatrix = None

def wordlen_filtering():
    global stopwords 
    global word_length_filter
    global stopWordsRun
    global worvalue
    global dirForIndexing
    global lenmatrix
    
    dirForIndexing = {}
    stopWordsRun = False
    stopwords = []
    word_length_filter = True
    
    #classified()
    naiive_bayes('wordlength-model.txt', 'wordlength-result.txt')
    word_length_filter = False
    
    worlen = pd.read_csv('use_wordlength-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    worlen.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    
    Matrix = np.array(worlen[['type', 'predic']])
    worlen = np.array(worlen[['type', 'predic', 'result']])
    
    worvalue = calculate_value(worlen)
    lenmatrix = generate_matrix(Matrix)
    
    
def start_run():
    base_model()
    stopword_filtering()
    wordlen_filtering()
    
    

    
val1 = None 
val5 = None 
val10 = None  
val15 = None
val20 = None

   
def infrequentword_filtering():
    global freq1
    global freq5
    global freq10
    global freq15
    global freq20
    
    global dirForIndexing
    
    global val1
    global val5
    global val10
    global val15
    global val20
    start = time.clock() 
    
    dirForIndexing = {}
    classified()
    freq1 = True
    naiive_bayes('freq1-model.txt', 'freq1-result.txt')
    freq1 = False
    
    fe1 = pd.read_csv('use_freq1-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    fe1.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    fe1 = np.array(fe1[['type', 'predic', 'result']])
    
    val1 = calculate_value(fe1)
    
    dirForIndexing = {}
    #classified()
    freq5 = True
    naiive_bayes('freq5-model.txt', 'freq5-result.txt')
    freq5 = False
    
    fe5 = pd.read_csv('use_freq5-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    fe5.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    fe5 = np.array(fe5[['type', 'predic', 'result']])
    
    val5 = calculate_value(fe5)
    
    dirForIndexing = {}
    #classified()
    freq10 = True
    naiive_bayes('freq10-model.txt', 'freq10-result.txt')
    freq10 = False
    
    fe10 = pd.read_csv('use_freq10-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    fe10.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    fe10 = np.array(fe10[['type', 'predic', 'result']])
    
    val10 = calculate_value(fe10)
    
    dirForIndexing = {}
    #classified()
    freq15 = True
    naiive_bayes('freq15-model.txt', 'freq15-result.txt')
    freq15 = False
    
    fe15 = pd.read_csv('use_freq15-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    fe15.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    fe15 = np.array(fe15[['type', 'predic', 'result']])
    val15 = calculate_value(fe15)
    
    dirForIndexing = {}
    #classified()
    freq20 = True
    naiive_bayes('freq20-model.txt', 'freq20-result.txt')
    freq20 = False
    
    fe20 = pd.read_csv('use_freq20-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    fe20.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    fe20 = np.array(fe20[['type', 'predic', 'result']])
    val20 = calculate_value(fe20)
    
    elapsed = (time.clock() - start)
    print('Time used to caculate four value is:', elapsed)
    
    
    plt.figure(figsize=(15,15))
    
    x = [0, 0.2, 0.4, 0.6, 0.8]
    my_xticks = np.arange(0, 1.1, 0.2)
    my_yticks = np.arange(0, 1.1, 0.2)
    #scale_ls = range(10)
    index_ls = ['Fq 1', 'Fq 5', 'Fq 10', 'Fq 15', 'Fq 20']
    
    plt.subplot(2,2,1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Accuracy')
    
    accuracy = [val1['accuracy'], val5['accuracy'], val10['accuracy'], val15['accuracy'], val20['accuracy']]

    #accur = [accuracy1, accuracy5, accuracy10, accuracy15, accuracy20]
    plt.plot(x, accuracy, color = 'black', markersize = 6, marker = 'o', label = 'total')
    plt.legend(loc='best')
    
    plt.subplot(2,2,2)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Precision')
    
    prestory = [val1['precision_story'], val5['precision_story'], val10['precision_story'], val15['precision_story'], val20['precision_story']]
    preask = [val1['precision_ask'], val5['precision_ask'], val10['precision_ask'], val15['precision_ask'], val20['precision_ask']]
    preshow = [val1['precision_show'], val5['precision_show'], val10['precision_show'], val15['precision_show'], val20['precision_show']]
    prepoll = [val1['precision_poll'], val5['precision_poll'], val10['precision_poll'], val15['precision_poll'], val20['precision_poll']]
    
    plt.plot(x, prestory, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, preask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, preshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, prepoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    
    plt.subplot(2,2,3)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Recall')
    
    recstory = [val1['recall_story'], val5['recall_story'], val10['recall_story'], val15['recall_story'], val20['recall_story']]
    recask = [val1['recall_ask'], val5['recall_ask'], val10['recall_ask'], val15['recall_ask'], val20['recall_ask']]
    recshow = [val1['recall_show'], val5['recall_show'], val10['recall_show'], val15['recall_show'], val20['recall_show']]
    recpoll = [val1['recall_poll'], val5['recall_poll'], val10['recall_poll'], val15['recall_poll'], val20['recall_poll']]
    
    plt.plot(x, recstory, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, recask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, recshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, recpoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    
    plt.subplot(2,2,4)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks,index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('F1-measure')
    
    f1story = [val1['F1_story'], val5['F1_story'], val10['F1_story'], val15['F1_story'], val20['F1_story']]
    f1cask = [val1['F1_ask'], val5['F1_ask'], val10['F1_ask'], val15['F1_ask'], val20['F1_ask']]
    f1cshow = [val1['F1_show'], val5['F1_show'], val10['F1_show'], val15['F1_show'], val20['F1_show']]
    f1cpoll = [val1['F1_poll'], val5['F1_poll'], val10['F1_poll'], val15['F1_poll'], val20['F1_poll']]
    
    plt.plot(x, f1story, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, f1cask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, f1cshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, f1cpoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    plt.show()


tval5 =None
tval10 =None
tval15 =None
tval20 =None
tval25 =None


def infrequentword_filtering_part2():
    global top5
    global top10
    global top15
    global top20
    global top25
    
    global dirForIndexing
    
    global tval5
    global tval10
    global tval15
    global tval20
    global tval25
    
    start = time.clock()
    
    dirForIndexing = {}
    classified()
    top5 = True
    naiive_bayes('top5-model.txt', 'top5-result.txt')
    top5 = False
    
    tp5 = pd.read_csv('use_top5-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    tp5.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    tp5 = np.array(tp5[['type', 'predic', 'result']])
    
    tval5 = calculate_value(tp5)
    
    #classified()
    dirForIndexing = {}
    top10 = True
    naiive_bayes('top10-model.txt', 'top10-result.txt')
    top10 = False
    
    tp10 = pd.read_csv('use_top10-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    tp10.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    tp10 = np.array(tp10[['type', 'predic', 'result']])
    
    tval10 = calculate_value(tp10)
    
    #classified()
    dirForIndexing = {}
    top15 = True
    naiive_bayes('top15-model.txt', 'top15-result.txt')
    top15 = False
    
    tp15 = pd.read_csv('use_top15-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    tp15.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    tp15 = np.array(tp15[['type', 'predic', 'result']])
    
    tval15 = calculate_value(tp15)
    
    #classified()
    dirForIndexing = {}
    top20 = True
    naiive_bayes('top20-model.txt', 'top20-result.txt')
    top20 = False
    
    tp20 = pd.read_csv('use_top20-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    tp20.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    tp20 = np.array(tp20[['type', 'predic', 'result']])
    
    tval20 = calculate_value(tp20)
    
    #classified()
    dirForIndexing = {}
    top25 = True
    naiive_bayes('top25-model.txt', 'top25-result.txt')
    top25 = False
    
    tp25 = pd.read_csv('use_top25-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    tp25.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    tp25 = np.array(tp25[['type', 'predic', 'result']])
    
    tval25 = calculate_value(tp25)
    
    elapsed = (time.clock() - start)
    print('Time used to caculate four value is:', elapsed)
    
    
    plt.figure(figsize=(15,15))
    
    x = [0, 0.2, 0.4, 0.6, 0.8]
    my_xticks = np.arange(0, 1.1, 0.2)
    my_yticks = np.arange(0, 1.1, 0.2)
    
    index_ls = ['Tp 5', 'Tp 10', 'Tp 15', 'Tp 20', 'Tp 25']
    
    plt.subplot(2,2,1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Accuracy')
    
    accuracy = [tval5['accuracy'], tval10['accuracy'], tval15['accuracy'], tval20['accuracy'], tval25['accuracy']]

    
    plt.plot(x, accuracy, color = 'black', markersize = 6, marker = 'o', label = 'total')
    plt.legend(loc='best')
    
    plt.subplot(2,2,2)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Precision')
    
    prestory = [tval5['precision_story'], tval10['precision_story'], tval15['precision_story'], tval20['precision_story'], tval25['precision_story']]
    preask = [tval5['precision_ask'], tval10['precision_ask'], tval15['precision_ask'], tval20['precision_ask'], tval25['precision_ask']]
    preshow = [tval5['precision_show'], tval10['precision_show'], tval15['precision_show'], tval20['precision_show'], tval25['precision_show']]
    prepoll = [tval5['precision_poll'], tval10['precision_poll'], tval15['precision_poll'], tval20['precision_poll'], tval25['precision_poll']]
    
    plt.plot(x, prestory, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, preask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, preshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, prepoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    
    plt.subplot(2,2,3)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Recall')
    
    recstory = [tval5['recall_story'], tval10['recall_story'], tval15['recall_story'], tval20['recall_story'], tval25['recall_story']]
    recask = [tval5['recall_ask'], tval10['recall_ask'], tval15['recall_ask'], tval20['recall_ask'], tval25['recall_ask']]
    recshow = [tval5['recall_show'], tval10['recall_show'], tval15['recall_show'], tval20['recall_show'], tval25['recall_show']]
    recpoll = [tval5['recall_poll'], tval10['recall_poll'], tval15['recall_poll'], tval20['recall_poll'], tval25['recall_poll']]
    
    plt.plot(x, recstory, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, recask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, recshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, recpoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    
    plt.subplot(2,2,4)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks,index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('F1-measure')
    
    f1story = [tval5['F1_story'], tval10['F1_story'], tval15['F1_story'], tval20['F1_story'], tval25['F1_story']]
    f1cask = [tval5['F1_ask'], tval10['F1_ask'], tval15['F1_ask'], tval20['F1_ask'], tval25['F1_ask']]
    f1cshow = [tval5['F1_show'], tval10['F1_show'], tval15['F1_show'], tval20['F1_show'], tval25['F1_show']]
    f1cpoll = [tval5['F1_poll'], tval10['F1_poll'], tval15['F1_poll'], tval20['F1_poll'], tval25['F1_poll']]
    
    plt.plot(x, f1story, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, f1cask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, f1cshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, f1cpoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    plt.show()

    
s1val = None
s2val = None 
s3val = None
s4val = None
s5val = None
s6val = None
s7val = None
s8val = None
s9val = None
s10val = None    

def smoothing():
    global smoothed
    global s1val
    global s2val
    global s3val
    global s4val
    global s5val
    global s6val
    global s7val
    global s8val
    global s9val
    global s10val
    
    global dirForIndexing
    
    dirForIndexing = {}
    smoothed = 0.1
    classified()
    naiive_bayes('smoothed0.1.txt', 'smoothed0.1-result.txt')
    
    s1 = pd.read_csv('use_smoothed0.1-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s1.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s1 = np.array(s1[['type', 'predic', 'result']])
    
    s1val = calculate_value(s1)
    
    
    dirForIndexing = {}
    smoothed = 0.2
    #classified()
    naiive_bayes('smoothed0.2.txt', 'smoothed0.2-result.txt')
    
    s2 = pd.read_csv('use_smoothed0.2-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s2.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s2 = np.array(s2[['type', 'predic', 'result']])
    
    s2val = calculate_value(s2)
    
    dirForIndexing = {}
    smoothed = 0.3
    #classified()
    naiive_bayes('smoothed0.3.txt', 'smoothed0.3-result.txt')
    
    s3 = pd.read_csv('use_smoothed0.3-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s3.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s3 = np.array(s3[['type', 'predic', 'result']])
    
    s3val = calculate_value(s3)
    
    dirForIndexing = {}
    smoothed = 0.4
    #classified()
    naiive_bayes('smoothed0.4.txt', 'smoothed0.4-result.txt')
    
    s4 = pd.read_csv('use_smoothed0.4-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s4.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s4 = np.array(s4[['type', 'predic', 'result']])
    
    s4val = calculate_value(s4)
    
    dirForIndexing = {}
    smoothed = 0.5
    #classified()
    naiive_bayes('smoothed0.5.txt', 'smoothed0.5-result.txt')
    
    s5 = pd.read_csv('use_smoothed0.5-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s5.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s5 = np.array(s5[['type', 'predic', 'result']])
    
    s5val = calculate_value(s5)
    
    dirForIndexing = {}
    smoothed = 0.6
    #classified()
    naiive_bayes('smoothed0.6.txt', 'smoothed0.6-result.txt')
    
    s6 = pd.read_csv('use_smoothed0.6-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s6.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s6 = np.array(s6[['type', 'predic', 'result']])
    
    s6val = calculate_value(s6)
    
    dirForIndexing = {}
    smoothed = 0.7
    #classified()
    naiive_bayes('smoothed0.7.txt', 'smoothed0.7-result.txt')
    
    s7 = pd.read_csv('use_smoothed0.7-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s7.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s7 = np.array(s7[['type', 'predic', 'result']])
    
    s7val = calculate_value(s7)
    
    dirForIndexing = {}
    smoothed = 0.8
    #classified()
    naiive_bayes('smoothed0.8.txt', 'smoothed0.8-result.txt')
    
    s8 = pd.read_csv('use_smoothed0.8-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s8.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s8 = np.array(s8[['type', 'predic', 'result']])
    
    s8val = calculate_value(s8)
    
    dirForIndexing = {}
    smoothed = 0.9
    #classified()
    naiive_bayes('smoothed0.9.txt', 'smoothed0.9-result.txt')
    
    s9 = pd.read_csv('use_smoothed0.9-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s9.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s9 = np.array(s9[['type', 'predic', 'result']])
    
    s9val = calculate_value(s9)
    
    dirForIndexing = {}
    smoothed = 1.0
    #classified()
    naiive_bayes('smoothed1.0.txt', 'smoothed1.0-result.txt')
    
    s10 = pd.read_csv('use_smoothed1.0-result.txt', header=None, sep = ',,,' , keep_default_na=False, na_values=['NA'])
    s10.columns = ['serial', 'title', 'type', 'score1','score2','score3', 'score4', 'predic', 'result']
    s10 = np.array(s10[['type', 'predic', 'result']])
    
    s10val = calculate_value(s10)
    
    
    
    plt.figure(figsize=(15,15))
    
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    my_xticks = np.arange(0, 1.1, 0.1)
    my_yticks = np.arange(0, 1.1, 0.2)
    #scale_ls = range(10)
    index_ls = ['Sm1', 'Sm2', 'Sm3', 'Sm4', 'Sm5', 'Sm6', 'Sm7', 'Sm8', 'Sm9', 'Sm10']
    
    plt.subplot(2,2,1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Accuracy')
    
    accuracy = [s1val['accuracy'], s2val['accuracy'], s3val['accuracy'], s4val['accuracy'], s5val['accuracy'], 
                s6val['accuracy'], s7val['accuracy'], s8val['accuracy'], s9val['accuracy'], s10val['accuracy']]

    #accur = [accuracy1, accuracy5, accuracy10, accuracy15, accuracy20]
    plt.plot(x, accuracy, color = 'black', markersize = 6, marker = 'o', label = 'total')
    plt.legend(loc='best')
    
    plt.subplot(2,2,2)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Precision')
    
    prestory = [s1val['precision_story'], s2val['precision_story'], s3val['precision_story'], s4val['precision_story'], s5val['precision_story'],
                s6val['precision_story'], s7val['precision_story'], s8val['precision_story'], s9val['precision_story'], s10val['precision_story']]
    preask = [s1val['precision_ask'], s2val['precision_ask'], s3val['precision_ask'], s4val['precision_ask'], s5val['precision_ask'],
              s6val['precision_ask'], s7val['precision_ask'], s8val['precision_ask'], s9val['precision_ask'], s10val['precision_ask']]
    preshow = [s1val['precision_show'], s2val['precision_show'], s3val['precision_show'], s4val['precision_show'], s5val['precision_show'],
               s6val['precision_show'], s7val['precision_show'], s8val['precision_show'], s9val['precision_show'], s10val['precision_show']]
    prepoll = [s1val['precision_poll'], s2val['precision_poll'], s3val['precision_poll'], s4val['precision_poll'], s5val['precision_poll'],
               s6val['precision_poll'], s7val['precision_poll'], s8val['precision_poll'], s9val['precision_poll'], s10val['precision_poll']]
    
    plt.plot(x, prestory, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, preask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, preshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, prepoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    
    plt.subplot(2,2,3)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks, index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('Recall')
    
    recstory = [s1val['recall_story'], s2val['recall_story'], s3val['recall_story'], s4val['recall_story'], s5val['recall_story'],
                s6val['recall_story'], s7val['recall_story'], s8val['recall_story'], s9val['recall_story'], s10val['recall_story']]
    recask = [s1val['recall_ask'], s2val['recall_ask'], s3val['recall_ask'], s4val['recall_ask'], s5val['recall_ask'],
              s6val['recall_ask'], s7val['recall_ask'], s8val['recall_ask'], s9val['recall_ask'], s10val['recall_ask']]
    recshow = [s1val['recall_show'], s2val['recall_show'], s3val['recall_show'], s4val['recall_show'], s5val['recall_show'],
               s6val['recall_show'], s7val['recall_show'], s8val['recall_show'], s9val['recall_show'], s10val['recall_show']]
    recpoll = [s1val['recall_poll'], s2val['recall_poll'], s3val['recall_poll'], s4val['recall_poll'], s5val['recall_poll'],
               s6val['recall_poll'], s7val['recall_poll'], s8val['recall_poll'], s9val['recall_poll'], s10val['recall_poll']]
    
    plt.plot(x, recstory, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, recask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, recshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, recpoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    
    plt.subplot(2,2,4)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(my_xticks,index_ls)
    plt.yticks(my_yticks)
    plt.ylabel('F1-measure')
    
    f1story = [s1val['F1_story'], s2val['F1_story'], s3val['F1_story'], s4val['F1_story'], s5val['F1_story'],
               s6val['F1_story'], s7val['F1_story'], s8val['F1_story'], s9val['F1_story'], s10val['F1_story']]
    f1cask = [s1val['F1_ask'], s2val['F1_ask'], s3val['F1_ask'], s4val['F1_ask'], s5val['F1_ask'],
              s6val['F1_ask'], s7val['F1_ask'], s8val['F1_ask'], s9val['F1_ask'], s10val['F1_ask']]
    f1cshow = [s1val['F1_show'], s2val['F1_show'], s3val['F1_show'], s4val['F1_show'], s5val['F1_show'],
               s6val['F1_show'], s7val['F1_show'], s8val['F1_show'], s9val['F1_show'], s10val['F1_show']]
    f1cpoll = [s1val['F1_poll'], s2val['F1_poll'], s3val['F1_poll'], s4val['F1_poll'], s5val['F1_poll'],
               s6val['F1_poll'], s7val['F1_poll'], s8val['F1_poll'], s9val['F1_poll'], s10val['F1_poll']]
    
    plt.plot(x, f1story, color = 'r', markersize = 6, marker = 'o', label = 'story')
    plt.plot(x, f1cask, color = 'b', markersize = 6, marker = 'o', label = 'ask_hn')
    plt.plot(x, f1cshow, color = 'g', markersize = 6, marker = 'o', label = 'show_hn')
    plt.plot(x, f1cpoll, color = 'y', markersize = 6, marker = 'o', label = 'poll')
    
    plt.legend(loc= 'best')
    
    plt.show()
    
    
    
    
    
