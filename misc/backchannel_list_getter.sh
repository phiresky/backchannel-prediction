set -eux
topcount=$1
head -$topcount dialog_act_corpus_stats_bywords_max3words.txt|cut -f5|sort|uniq|
    sed "/^$/d" | sed "s/ '/'/g" | sed "s/ n't/n't/g" >../data/backchannels-top$topcount.txt
