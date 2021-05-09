Project 2 Program instruction

Project 2 is a big part which contains 3 tasks, and the last task in separate as 4 subtasks.

So, in order to show each task results step by step, I did't packed the whole module.

Step1, to run task 1 and task 2 together(since part1 is training and part2 is testing) and check the result, 
please run the "base_model()" function,
and after that it will create "model-2018.txt", "baseline-result.txt" and "vocabulary.txt" in local.
To see the results of the exp. #1, there is a global variable "basvalue",
to see the accuracy, recall, precision, F1-measure for each class.

step2, to run exp. #2, please run "stopword_filtering()" function,
it will create "stopword-model.txt", "stopword-result.txt".
To see results, check global variable "stopvalue", same as above.
 
step3, to run exp. #3, please run "wordlen_filtering()" function,
it will create "wordlength-model.txt", "wordlength-result.txt".
To see results, check global variable "worvalue", same as above.

step4, to run exp. #4, I split two part to see frequency influent run "infrequentword_filtering()" function,
it will plot performance for each class,
to see result, there are five global variables,
"val1", "val5", "val10", "val15", "val20".   each represent the frequency that removed

part2: to see the top % performance, run "infrequentword_filtering_part2()" function,
it will plot performance for each class,
to see result, there are five global variables,
"tval5", "tval10", "tval15", "tval20", "tval25". each represent the top % that been removed.


step 5 to run exp. #4, please run "smoothing()" function,
it will plot performance for each class during 10 test.
to see result, there are ten global variables,
"s1val", "s2val", "s3val", "s4val", "s5val", "s6val", "s7val", "s8val", "s9val", "s10val"








 