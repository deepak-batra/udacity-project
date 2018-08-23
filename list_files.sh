#!/bin/bash
for entry in capstone_images/train/*
do
  echo "$entry," `ls -l "$entry"/ | wc -l`
done

