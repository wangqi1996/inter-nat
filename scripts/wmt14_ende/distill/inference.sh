DISTILL=/home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq/
SAVEDIR=/home/wangdq/save/inter/wmt14_ende_distill/$4
log=$2
genset=$3

export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
fairseq-generate \
  $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat3 \
  --gen-subset $genset \
  --seed 1234 \
  --task nat \
  --remove-bpe \
  --batch-size 128 \
  --beam 1 \
  --remove-bpe \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  -s en -t de \
  --path $SAVEDIR/checkpoint_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path ~/$log \
  --model-overrides "{'valid_subset': '$genset'}" \
  --left-pad-source False

tail -1 ~/$log/generate-$genset.txt

python scripts/average_checkpoints.py --input $SAVEDIR/checkpoint.best --output $SAVEDIR/checkpoint_ave_best.pt

fairseq-generate \
  $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat3 \
  --gen-subset $genset \
  --seed 1234 \
  --task nat \
  --remove-bpe \
  --batch-size 128 \
  --beam 1 \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  -s en -t de \
  --path $SAVEDIR/checkpoint_ave_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path ~/$log \
  --model-overrides "{'valid_subset': '$genset'}" \
  --left-pad-source False

tail -1 ~/$log/generate-$genset.txt
