#!/usr/bin/env bash

table_name="research_project:dkuzin.kmean_results"
source="gs://dkuzin/experiments/tensorflow/kmean-sites/part-*"

time bq load --null_marker "" --skip_leading_rows=0 --max_bad_records=0 --field_delimiter='\t' $table_name $source table_schema.json

#time bq load --null_marker "" --max_bad_records=0 --field_delimiter='\t' --quote="" $table_name $source "site:string,num:integer,cluster:integer"





