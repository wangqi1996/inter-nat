DISTILL=/home/data_ti5_c/wangdq/data/fairseq/iwslt14/deen-AT/
export CUDA_VISIBLE_DEVICES=$1

fairseq-generate $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat \
  --gen-subset $3 \
  --seed 1234 \
  --task nat \
  --remove-bpe \
  --batch-size 16 \
  --beam 1 \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  -s de -t en \
  --path /home/wangdq/save/inter/iwslt14_deen_distill/$4/checkpoint_ave_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path ~/$2 \
  --model-overrides "{'valid_subset': '$3'}" \
  --left-pad-source False \
  --infer-with-reflen \
  --use-oracle-mat

tail -1 ~/$2/generate-$3.txt
