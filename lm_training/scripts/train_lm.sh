dir_path=$1
lang=$2
#top_k=500000
#lexicon=500000
top_k=10000
lexicon=10000

#For kenlm
lm_path=${dir_path}"/"${lang}
kenlm_bins="/home/ubuntu/kenlm/build/bin" #path to kenlm binary
input_txt_file_path=${lm_path}"/ling_sent.txt"
#input_txt_file_path=${lm_path}"/swc_sent.txt" #LM_small for SWC

#For lexicon 
lexicon_vocab_file=${lm_path}"/lexicon.txt"
path_to_save_lexicon=${out_dir}"/"${lang}"/lexicon.lst"

#Previous

#printf "\n** Generating kenlm **\n"
#python utils/train_lm.py --input_txt ${input_txt_file_path} --lm_dir ${lm_path} \
#    --lexicon ${lexicon} --top_k ${top_k} --kenlm_bins ${kenlm_bins} \
#    --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
#    --binary_a_bits 255 --binary_q_bits 8 --binary_type trie 
#printf "**Kenlm Generated at : "${lm_path}

#new
printf "\n** Generating kenlm **\n"
python utils/train_lm.py --input_txt ${input_txt_file_path} --lm_dir ${lm_path} \
    --lexicon ${lexicon} --top_k ${top_k} --kenlm_bins ${kenlm_bins} \
    --arpa_order 6 --max_arpa_memory "90%" --arpa_prune "0|0|0|0|1|2" \
    --binary_a_bits 255 --binary_q_bits 8 --binary_type trie 
printf "**Kenlm Generated at : "${lm_path}