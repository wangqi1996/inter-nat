dir="/home/wangdq/latent_glat_fast_align/distill-output-word"
source="source.txt"
target="word.txt"
result="source-word"
align_output="train-align"
python /home/data_ti5_c/wangdq/code/dep_nat/toy/Cod/process.py --dir $dir --source $source --target $target --result $result

wait
fast_align_dir="/home/data_ti5_c/wangdq/code/fast_align"
$fast_align_dir/build/fast_align -i $result -d -o -v -p fwd_params >fwd_align 2>fwd_err
wait
$fast_align_dir/build/fast_align -i $result -d -o -v -r -p rev_params >rev_align 2>rev_err
wait
python2 $fast_align_dir/build/force_align.py fwd_params fwd_err rev_params rev_err grow-diag-final-and <$result >$align_output
wait
python /home/data_ti5_c/wangdq/code/dep_nat/toy/Cod/compute_cod.py --dir $dir --source $source --target $target --align $align_output
