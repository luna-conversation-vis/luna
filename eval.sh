#!/bin/bash

for split in "val" "test" 
do 
    for v_num in 0
    do 
        echo "======================== run $v_num: $split ========================"
        python pipeline.py $split --select_count=$v_num --select_col=$v_num --where_col=$v_num --where_count=$v_num --where_operator=$v_num --where_value=$v_num --verbosity=0
    done
done