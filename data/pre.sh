#!/bin/bash -e
tr '\n' ' ' \
| sed "s/\t/ /g" \
| sed "s/(\([^()]*\))/\n\1\n/g" \
| sed "s/\"\([^\"]*\)\"/\n\1\n/g" \
| sed "s/\'\([^\']*\)\'/\n\1\n/g" \
| sed "s/\[[^][]*\]//g" \
| sed "s/[,:\/\` ]/ /g" \
| sed "s/[\?\!\.;]/\n/g" \
| sed "s/[^a-zA-Z0-9 ]/ /g" \
| sed "s/./\L&/g" \
| sed "s/ [ ]*/ /g" \
| sed "s/^[ \t]*//g" \
| sed "s/[\t ]*$//g" \
| sed "/^$/d" \
| sed "/^[^ ]*$/d" #\
#| sed "s/^/<s> /" \
#| sed "s/$/ <\/s>/" 

#| sed "s/' / /g" \
#| sed "s/ '/ /g" \
#| sed "s/ -/ /g" \
#| sed "s/- / /g" \
