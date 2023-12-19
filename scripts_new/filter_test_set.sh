 for i in {0..7}; do
 sh ./scripts_new/filter_lines.sh ../../../data/parallel/test/en_ids.txt ../../../data/parallel/test/test.target.$i > ../../../data/parallel/test/test.en.$i
 done