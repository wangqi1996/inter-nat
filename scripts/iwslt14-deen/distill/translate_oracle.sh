DISTILL=/home/data_ti5_c/wangdq/data/fairseq/iwslt14/deen-AT/
export CUDA_VISIBLE_DEVICES=$1
log=$2
genset=$3
path=/home/wangdq/save/inter/iwslt14_deen_distill/$4

fairseq-generate $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat4 \
  --gen-subset $genset \
  --seed 1234 \
  --task nat \
  --remove-bpe \
  --batch-size 16 \
  --beam 1 \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  -s de -t en \
  --path $path/checkpoint_ave_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path ~/$log \
  --model-overrides "{'valid_subset': '$genset'}" \
  --left-pad-source False \
  --infer-with-reflen \
  --use-oracle-mat

tail -1 ~/$log/generate-$genset.txt
