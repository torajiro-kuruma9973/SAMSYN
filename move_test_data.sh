#!/bin/bash

rm -f samsyn_dataset/test_data/*
rm -f samsyn_dataset/test_labels/*
rm -f samsyn_dataset/test_segs/*

cp samsyn_datasets_backup/data_backup/${1}.nii.gz samsyn_dataset/test_data/
cp samsyn_datasets_backup/labels_backup/${1}.nii.gz samsyn_dataset/test_labels/
cp samsyn_datasets_backup/segs_backup/${1}.nii.gz samsyn_dataset/test_segs/

echo "✅ 任务完成！"